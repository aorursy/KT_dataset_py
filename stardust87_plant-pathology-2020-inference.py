!pip install timm
import os,time



import numpy as np

import pandas as pd



import albumentations as A

import cv2



import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torch.optim as optim



import timm



from tqdm.notebook import tqdm

from torch.utils.data import Dataset, DataLoader

from albumentations.pytorch import ToTensorV2



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split





import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



import warnings  

warnings.filterwarnings('ignore')



DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'

IMAGE_INPUT = '/kaggle/input/plant-pathology-2020-resized-images'



SEED = 42

N_FOLDS = 5

N_EPOCHS = 20

BATCH_SIZE = 8

IMAGE_SIZE = (409,273)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = torch.device(device)

device
class PlantDataset(Dataset):

    

    def __init__(self, df, transforms=None, test_set=False):

    

        self.df = df

        self.transforms = transforms

        self.test_set = test_set

        if not self.transforms:

            self.transforms  = A.Compose([ToTensorV2(p=1.0)])

        

    def __len__(self):

        return self.df.shape[0]

    

    def __getitem__(self, idx):

        image_src = IMAGE_INPUT + '/images_409_273/' + self.df.loc[idx, 'image_id'] + '.jpg'

        image = cv2.imread(image_src, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        transformed = self.transforms(image=image)

        image = transformed['image']

        

        if not self.test_set:

            labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values

            labels = torch.from_numpy(labels.astype(np.int8))

            labels = labels.unsqueeze(-1)

                

            return image, labels

        else:

            return image

def trim_network_at_index(network,index=-1):

    assert index <0, f'Param index must be negative. Received {index}'

    return nn.Sequential(*list(network.children())[:index])
class PlantModel(nn.Module):

    

    def __init__(self, num_classes=4):

        super().__init__()

        self.backbone = timm.create_model('resnest269e',pretrained=True)

#         self.backbone = torchvision.models.resnet50(pretrained=True)

#         self.backbone = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)

#         in_features = self.backbone.classifier.in_features

        in_features = self.backbone.fc.in_features

        self.backbone = trim_network_at_index(self.backbone,-1) 

        self.logit = nn.Linear(in_features, num_classes)

        

    def forward(self, x):

        x = self.backbone(x).flatten(start_dim=1)



        x = self.logit(x)



        return  x
# class PlantModel(nn.Module):

    

#     def __init__(self, num_classes=4):

#         super().__init__()

        

#         self.backbone = torchvision.models.resnet50(pretrained=True)

        

#         in_features = self.backbone.fc.in_features



#         self.logit = nn.Linear(in_features, num_classes)

        

#     def forward(self, x):

#         batch_size, C, H, W = x.shape

        

#         x = self.backbone.conv1(x)

#         x = self.backbone.bn1(x)

#         x = self.backbone.relu(x)

#         x = self.backbone.maxpool(x)



#         x = self.backbone.layer1(x)

#         x = self.backbone.layer2(x)

#         x = self.backbone.layer3(x)

#         x = self.backbone.layer4(x)

        

#         x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)

#         x = F.dropout(x, 0.25, self.training)



#         x = self.logit(x)



#         return x
transforms_valid = A.Compose([

    A.Normalize(p=1.0),

    ToTensorV2(p=1.0)

])



test_df = pd.read_csv(DIR_INPUT + '/test.csv')

dataset_test = PlantDataset(df=test_df,test_set = True, transforms=transforms_valid)

testloader = DataLoader(dataset_test, batch_size=BATCH_SIZE,

                                          shuffle=False, num_workers=4)
def test_model(model_name,testloader):

    model_path = f'/kaggle/input/plant-pathology-2020-training/{model_name}.pth'

    model = PlantModel()

    model.load_state_dict(torch.load(model_path,map_location=device))

    model = model.to(device)

    

    test_probs = []

    model.eval()



    test_iter = iter(testloader)



    with torch.no_grad():

        for i in tqdm(range(len(testloader))):        

            image = next(test_iter)

            probs = F.softmax(model(image.to(device)))

            test_probs.append(probs.cpu().numpy())





    test_probs = np.concatenate(test_probs, axis=0)

    

    return test_probs
test_probs = []

start = time.perf_counter()

for i_fold in range(N_FOLDS):

    test_probs_fold = test_model(f'modelF{i_fold}',testloader)

    test_probs.append(test_probs_fold)

print(f'Finished Inference in {(time.perf_counter() - start):.2f} seconds')
test_probs_mean = [np.mean(k,axis=0) for k in zip(*test_probs)]

submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')

submission_df.iloc[:, 1:] = 0

submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = test_probs_mean

submission_df.to_csv('submission.csv', index=False)
submission_df