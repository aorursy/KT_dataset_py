# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import torch.nn.functional as F

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy

from sklearn.metrics import confusion_matrix

from torch.utils.data import random_split, Dataset

from torch.utils.data.sampler import SubsetRandomSampler

import glob

from PIL import Image

import glob

import cv2

import gc #garbage collector for gpu memory 
transform = transforms.Compose(

        [transforms.Resize((320,320)),

        transforms.ToTensor(),

        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081])

        ])



class APTOSDataset(Dataset):

    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):

        self.eye_frame = pd.read_csv(csv_file)

        self.filetype = filetype

        self.transform = transform



    def __len__(self):

        return len(self.eye_frame)

    

    def __getitem__(self, idx):

        if self.filetype == 'train':

            img_name = os.path.join('../input/aptos2019-blindness-detection/train_images',

                                    self.eye_frame.loc[idx,'id_code'] + '.png')



            image = Image.open(img_name)

            if self.transform:

                image = self.transform(image)

            else:

                image = transforms.ToTensor()(image)



            return image,self.eye_frame.diagnosis[idx]

        

        else:

            img_name = os.path.join('../input/aptos2019-blindness-detection/test_images',

                                    self.eye_frame.loc[idx,'id_code'] + '.png')

            image = Image.open(img_name)

            if self.transform:

                image = self.transform(image)

            else:

                image = transforms.ToTensor()(image)

            return image, self.eye_frame.loc[idx,'id_code']
test_dataset = APTOSDataset(csv_file='../input/aptos2019-blindness-detection/test.csv', filetype='test',transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=4)





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





model0 = torchvision.models.resnet152(pretrained=False)

state = torch.load('../input/resnet/FinalResnet152_0.pt')

num_ftrs = model0.fc.in_features

model0.fc = nn.Linear(num_ftrs,5)

model0.load_state_dict(state)

model0 = model0.to(device)



model1 = torchvision.models.resnet101(pretrained=False)

state = torch.load('../input/resnet/FinalResnet02.pt')

num_ftrs = model1.fc.in_features

model1.fc = nn.Linear(num_ftrs,5)

model1.load_state_dict(state)

model1 = model1.to(device)



model2 = torchvision.models.resnet101(pretrained=False)

state = torch.load('../input/resnet/FinalResnet01.pt')

num_ftrs = model2.fc.in_features

model2.fc = nn.Linear(num_ftrs,5)

model2.load_state_dict(state)

model2 = model2.to(device)



model3 = torchvision.models.resnet101(pretrained=False)

state = torch.load('../input/resnet/FinalResnet00.pt')

num_ftrs = model3.fc.in_features

model3.fc = nn.Linear(num_ftrs,5)

model3.load_state_dict(state)

model3 = model3.to(device)



model4 = torchvision.models.resnet101(pretrained=False)

state = torch.load('../input/resnet/FinalResnet0.pt')

num_ftrs = model4.fc.in_features

model4.fc = nn.Linear(num_ftrs,5)

model4.load_state_dict(state)

model4 = model4.to(device)

from tqdm._tqdm_notebook import tqdm_notebook

def compute_predictions(model, model_type, data_loader, device):

    if model_type == 'train':

        predictions = []

        correct_pred, num_examples = 0, 0

        tqdm_notebook()

        for i, (inputs, labels) in enumerate(tqdm_notebook(data_loader)):

            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            predictions.append(preds)

            num_examples += labels.size(0)

            correct_pred += (preds==labels).sum()

        return predictions, correct_pred.item()/num_examples*100

    

    else:

        predictions = []

        img_ids = []

        out = []

        tqdm_notebook()



        for i, (inputs, img_id) in enumerate(tqdm_notebook(data_loader)):

            inputs = inputs.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            predictions.extend(preds)

            img_ids.extend(img_id)

            out.extend(outputs)

            

        predictions = [pred.item() for pred in predictions]

        final_predictions = pd.DataFrame(np.array([img_ids, predictions])).transpose()

        final_predictions.columns = ['id_code', 'diagnosis']

        final_predictions_df = final_predictions.copy()

        #final_predictions.to_csv("submission.csv",index=False)

        return final_predictions_df, out, img_ids
with torch.set_grad_enabled(False):

    model0.eval()

    model1.eval()

    model2.eval()

    model3.eval()

    model4.eval()



    #train_predictions, train_accuracy = compute_predictions(model, 'train', train_loader, device)

    #print('Train Accuracy: ', train_accuracy)

    print('Computing Test Predictions')

    test_predictions0,out0,ids0 = compute_predictions(model0, 'test', test_loader, device)

    test_predictions1,out1,ids1 = compute_predictions(model1, 'test', test_loader, device)

    test_predictions2,out2,ids2 = compute_predictions(model2, 'test', test_loader, device)

    test_predictions3,out3,ids3 = compute_predictions(model3, 'test', test_loader, device)

    test_predictions4,out4,ids4 = compute_predictions(model4, 'test', test_loader, device)



out = (torch.stack(out0)+torch.stack(out1)+torch.stack(out2)+torch.stack(out3)+torch.stack(out4))/5

img_ids = np.array(ids0)

_, predictions = torch.max(out, 1)

predictions = predictions.cpu().numpy()

final_predictions = pd.DataFrame(np.array([img_ids, predictions])).transpose()

final_predictions.columns = ['id_code', 'diagnosis']

final_predictions_df = final_predictions.copy()

final_predictions.to_csv("submission.csv",index=False)