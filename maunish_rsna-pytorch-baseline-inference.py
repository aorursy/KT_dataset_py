import random

import os

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.functional as F

import torch.optim as optim

import torchvision

from torchvision import models

from torch.utils.data import Dataset,DataLoader

import cv2

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



from collections import Counter



from sklearn.model_selection import KFold



import vtk

from vtk.util import numpy_support

from tqdm.auto import tqdm

folder_path = "../input/rsna-str-pulmonary-embolism-detection"

train_path = folder_path + "/train/"

test_path = folder_path + "/test/"

    

# train_data = pd.read_csv(folder_path + "/train.csv")

test_data  = pd.read_csv(folder_path + "/test.csv")

sample = pd.read_csv(folder_path + "/sample_submission.csv")



cols_ID = ["StudyInstanceUID","SeriesInstanceUID","SOPInstanceUID"]

test_data["ImagePath"] = test_path+ test_data[cols_ID[0]]+"/"+test_data[cols_ID[1]]+"/"+test_data[cols_ID[2]]+".dcm"
SEED  = 42



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASSEED']  = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)

target_columns = ['pe_present_on_image', 'negative_exam_for_pe', 'rv_lv_ratio_gte_1', 

                  'rv_lv_ratio_lt_1','leftsided_pe', 'chronic_pe','rightsided_pe', 

                  'acute_and_chronic_pe', 'central_pe', 'indeterminate']



study_level_columns = [ 'negative_exam_for_pe', 'rv_lv_ratio_gte_1', 

                  'rv_lv_ratio_lt_1','leftsided_pe', 'chronic_pe','rightsided_pe', 

                  'acute_and_chronic_pe', 'central_pe', 'indeterminate']



classes = len(target_columns)

model = models.resnet18(pretrained=False)

in_features = model.fc.in_features

model.fc = nn.Linear(in_features,classes)



model_path = "../input/rsna-super-cool-eda-and-pytorch-baseline-train/"



config={

       "learning_rate":0.001,

       "train_batch_size":32,

        "valid_batch_size":32,

        "test_batch_size":64,

       "epochs":10,

       "nfolds":3,

       "number_of_samples":7000

       }



reader = vtk.vtkDICOMImageReader()

def get_img(path):

    reader.SetFileName(path)

    reader.Update()

    _extent = reader.GetDataExtent()

    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]



    ConstPixelSpacing = reader.GetPixelSpacing()

    imageData = reader.GetOutput()

    pointData = imageData.GetPointData()

    arrayData = pointData.GetArray(0)

    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)

    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')

    ArrayDicom = cv2.resize(ArrayDicom,(512,512))

    return ArrayDicom





def convert_to_rgb(array):

    array = array.reshape((512, 512, 1))

    return np.stack([array, array, array], axis=2).reshape((3,512, 512))
class RsnaDataset(Dataset):

    

    def __init__(self,df,transforms=None):

        super().__init__()

        self.image_paths = df['ImagePath'].unique()

        self.df = df

        self.study_ids= df[cols_ID[1]].values

        self.sop_ids = df[cols_ID[2]].values

        self.transforms = transforms

    

    def __getitem__(self,index):

        

        image_path = self.image_paths[index]

        image = get_img(image_path)

        image = convert_to_rgb(image)

        

        study_id = self.study_ids[index]

        sop_id = self.sop_ids[index]

        

        if self.transforms:

            image = self.transforms(image=image)['image']

        



        image = torch.tensor(image,dtype=torch.float)        

        

        return {"image":image,

                "study_id":study_id,

                "sop_id":sop_id}   

    

    def __len__(self):

        return self.image_paths.shape[0]  
def inference():

    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_prediction = np.zeros((test_data.shape[0],len(target_columns)))

    study_ids = list()

    sop_ids = list()

    for i in range(config["nfolds"]):

        model.load_state_dict(torch.load(f"{model_path}model{i}.bin"))

        predictions = list()

        model.to(device)

        test_ds = RsnaDataset(test_data)

        test_dl = DataLoader(test_ds,

                        batch_size=config['test_batch_size'],

                        shuffle=False)

        

        tqdm_loader = tqdm(test_dl)

        

        with torch.no_grad():

            for inputs in tqdm_loader:

                images = inputs["image"].to(device, dtype=torch.float)

                outputs = model(images) 

                predictions.extend(outputs.cpu().detach().numpy())

                if i == 0:

                    study_ids.extend(inputs["study_id"])

                    sop_ids.extend(inputs["sop_id"])



        all_prediction += np.array(predictions)/config['nfolds']

        

    return all_prediction, study_ids, sop_ids
predictions,study_ids,sop_ids = inference()
df = pd.DataFrame(predictions)

df.columns = target_columns

df["StudyInstanceUID"] = study_ids

df["SOPInstanceUID"] = sop_ids
temp1 = df.groupby("StudyInstanceUID")[target_columns].mean().reset_index()

temp1.drop("pe_present_on_image",inplace=True,axis=1)

temp1 = pd.melt(temp1, id_vars=["StudyInstanceUID"], value_vars=study_level_columns)

temp1["label"] = temp1["StudyInstanceUID"].astype(str) + "_" +temp1["variable"].astype(str)

temp1.drop(["StudyInstanceUID","variable"],axis=1,inplace=True)

temp1.columns = ["label","id"]



temp2 = df.drop(study_level_columns +["StudyInstanceUID"],axis=1)

temp2 = pd.melt(temp2, id_vars=["SOPInstanceUID"], value_vars=['pe_present_on_image'])

temp2["label"] = temp2["SOPInstanceUID"].astype(str) 

temp2.drop(["SOPInstanceUID","variable"],axis=1,inplace=True)

temp2.columns = ["label","id"]



submission = temp2.append(temp1)

submission.to_csv("submission.csv",index=False)
Counter(sample.id) == Counter(submission.id)
print(submission.shape)

submission.head()