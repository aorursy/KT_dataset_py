import os

import pandas as pd

import numpy as np 



from PIL import Image



import torch

import torch.nn as nn

import torchvision

from torch.utils.data import Dataset

from torchvision import transforms



import xgboost as xgb

from sklearn.metrics import cohen_kappa_score



import pickle



DEVICE = torch.device("cuda:0")

DATA_SOURCE = os.path.join("..","input","aptos2019-blindness-detection")

MODEL_SOURCE = os.path.join("..","input","aptos-cnn-features-extraction-xgb-baseline")

# DATA_SOURCE = os.path.join("..","input")
class RetinopathyDatasetTest(Dataset):



    def __init__(self, eval_set=False, random_state=42):

        # read data list, split in train and eval, select the set

        csv_file = os.path.join(DATA_SOURCE, "test.csv")

        df = pd.read_csv(csv_file)

        self.data = df.reset_index(drop=True)

            

    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        # get image and process it to tensor ready for the model, extract features

        folder = os.path.join(DATA_SOURCE, "test_images")

        code = str(self.data.loc[idx, 'id_code'])

        file = code + ".png"

        path = os.path.join(folder, file)

        imgpil = Image.open(path)

        base_transforms = transforms.Compose([transforms.Resize((224, 224)),

                                              transforms.ToTensor(),

                                              transforms.Normalize([0.485, 0.456, 0.406], 

                                                                   [0.229, 0.224, 0.225])])

        img_tensor = base_transforms(imgpil)

        return {'image': img_tensor}
# load the pretrained CNN used as feature extractor

# no classifier defined, we will take the raw output from the CNN layers

extractor = torchvision.models.resnet101(pretrained=False)

extractor.fc = nn.Identity() 

model_path = os.path.join(MODEL_SOURCE, "resnet101.pth")

extractor.load_state_dict(torch.load(model_path))

extractor.to(DEVICE)

extractor.eval()



data_loader = torch.utils.data.DataLoader(RetinopathyDatasetTest(), 

                            batch_size=2, shuffle=False, num_workers=0, drop_last=False)



def get_extracted_data(data_loader):

    for bi, d in enumerate(data_loader):

        if bi % 32 == 0 : print(".", end="")

        img_tensor = d["image"].to(DEVICE)

        with torch.no_grad(): feature = extractor(img_tensor)

        feature = feature.cpu().detach().squeeze(0).numpy()

        if bi == 0 :

            features = feature 

        else :

            features = np.concatenate([features, feature], axis=0)

    print("")

    return features



print("...............................")

features = get_extracted_data(data_loader)
XGBOOST_PARAM = {

    "random_state" : 42,

    'objective': 'multi:softmax',

    "num_class" : 5,

    "n_estimators" : 200,

    "eval_metric" : "mlogloss"

}

xgb_model_1 = xgb.XGBClassifier(**XGBOOST_PARAM)

model_path = os.path.join(MODEL_SOURCE, "xgb_model_1")

xgb_model_1 = pickle.load(open(model_path, "rb"))

prediction1 = xgb_model_1.predict_proba(features)
xgb_model_2 = xgb.XGBClassifier(**XGBOOST_PARAM)

model_path = os.path.join(MODEL_SOURCE, "xgb_model_2")

xgb_model_2 = pickle.load(open(model_path, "rb"))

prediction2 = xgb_model_2.predict_proba(features)
prediction = (prediction1 + prediction2).argmax(axis=1)

csv_file = os.path.join(DATA_SOURCE, "sample_submission.csv")

df = pd.read_csv(csv_file)

df["diagnosis"] = prediction

df.to_csv('submission.csv',index=False)