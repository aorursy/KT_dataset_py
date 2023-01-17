import sys

import numpy as np

import pandas as pd

import torchvision

import torch.nn as nn

from tqdm import tqdm

from PIL import Image, ImageFile

from torch.utils.data import Dataset

import torch

from torchvision import transforms

import os



#package_dir = "../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/"

#sys.path.insert(0, package_dir)





device = torch.device('cuda:0')

ImageFile.LOAD_TRUNCATED_IMAGES = True
!pip install pretrainedmodels
import pretrainedmodels
class RetinopathyDatasetTest(Dataset):

    def __init__(self, csv_file, transform):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx,'id_code']+'.png')

        image = Image.open(img_name)

        image = self.transform(image)

        return {'image':image}
model = pretrainedmodels.__dict__['resnet101'](pretrained = None)

model.avg_pool = nn.AdaptiveAvgPool2d(1)

model.last_linear = nn.Sequential(

                        nn.BatchNorm1d(2048, eps =1e-5, momentum = 0.1, affine= True, track_running_stats = True),

                        nn.Dropout(0.25),

                        nn.Linear(in_features = 2048, out_features = 2048, bias = True),

                        nn.ReLU(),

                        nn.BatchNorm1d(2048, eps = 1e-5, momentum =0.1, affine = True, track_running_stats = True ),

                        nn.Dropout(0.5),

                        nn.Linear(in_features = 2048, out_features = 1, bias = True )

                        )



model.load_state_dict(torch.load("../input/mmmodel/model.bin"))

model = model.to(device)
for param in model.parameters():

    param.requires_grad = False

    

model.eval()
test_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    ])



test_dataset = RetinopathyDatasetTest("../input/aptos2019-blindness-detection/sample_submission.csv", transform = test_transform)
test_preds_all = np.zeros((len(test_dataset),10))

for j in range(10):

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = False, num_workers = 4)

    test_preds = np.zeros((len(test_dataset),1))

    tk0 = tqdm(test_data_loader)

    for i, x_batch in enumerate(tk0):

        x_batch = x_batch['image']

        pred = model(x_batch.to(device))

        test_preds[i*32:(i+1)*32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1,1)

    test_preds = test_preds.flatten()

    test_preds_all[:,j] = test_preds
test_preds_all
# Average the results

test_preds_agg = np.sum(test_preds_all,axis = 1)/10
test_preds_agg
coef = [0.5,1.5,2.5,3.5]



for i, pred in enumerate(test_preds_agg):

    if pred<coef[0]:

        test_preds_agg[i] = 0

    elif pred>=coef[0] and pred <coef[1]:

        test_preds_agg[i] = 1

    elif pred>=coef[1] and pred<coef[2]:

        test_preds_agg[i] = 2

    elif pred>=coef[2] and pred<coef[3]:

        test_preds_agg[i] = 3

    else:

        test_preds_agg[i] = 4

        



sample = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample.diagnosis = test_preds_agg.astype(int)

sample.to_csv('submission.csv',index = False)
