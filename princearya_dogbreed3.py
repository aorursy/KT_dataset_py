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
import cv2

import matplotlib.pyplot as plt

from os.path import isfile

import torch.nn.init as init

import torch

import torch.nn as nn

import numpy as np

import pandas as pd 

import os

from PIL import Image, ImageFilter

print(os.listdir("../input"))

from sklearn.model_selection import train_test_split, StratifiedKFold

from torch.utils.data import Dataset

from torchvision import transforms

from torch.optim import Adam, SGD, RMSprop

import time

from torch.autograd import Variable

import torch.functional as F

from tqdm import tqdm

from sklearn import metrics

import urllib

import pickle

import cv2

import torch.nn.functional as F

from torchvision import models

import seaborn as sns

import random

import sys

import albumentations

import albumentations.pytorch as AT

import numpy as np

import pandas as pd

import cv2

import matplotlib.image as mpimg

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, Dataset

from PIL import Image

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
!pip install albumentations > /dev/null
train_df = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
#train_df["id"]=train_df["id"].apply(lambda x : x+".jpg")
def prepare_labels(y):

    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter

    values = np.array(y)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)



    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)



    y = integer_encoded

    return y, label_encoder

y, lab_encoder = prepare_labels(train_df['breed'])
train_df["breed"]=y
test_df = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
train_df.head()
data_dir = '/kaggle/input/dog-breed-identification/'

train_dir =  '/kaggle/input/dog-breed-identification/train/'

test_dir =  '/kaggle/input/dog-breed-identification/test/'

labels = train_df
labels.head()
plt.figure(figsize=[15,15])

i = 1

for img_name in labels['id'][:10]:

    img = Image.open(train_dir + img_name + '.jpg')

    plt.subplot(6,5,i)

    plt.imshow(img)

    i += 1

plt.show()
class ImageData(Dataset):

    def __init__(self, df, data_dir, transform):

        super().__init__()

        self.df = df

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):       

        img_name = self.df.id[index] + '.jpg'

        label = self.df.breed[index]          

        img_path = os.path.join(self.data_dir, img_name)   

            

        image = mpimg.imread(img_path)

        image = (image + 1) * 127.5

        image = image.astype(np.uint8)

        

        image = self.transform(image)

        return image, label
data_transf = transforms.Compose([transforms.ToPILImage(mode='RGB'), 

                                  transforms.Resize(265),

                                  transforms.CenterCrop(224),

                                  transforms.ToTensor()])

train_data = ImageData(df = labels, data_dir = train_dir, transform = data_transf)

train_loader = DataLoader(dataset = train_data, batch_size=32, drop_last=True)
data = iter(train_loader)

model = models.resnet50()

model.load_state_dict(torch.load("/kaggle/input/pretrained-pytorch-models/resnet50-19c8e357.pth"))
for param in model.parameters():

    param.requires_grad = False
model.fc = nn.Linear(2048, 120)
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

loss_func = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2, )
%%time

# Train model

loss_log=[]

for epoch in range(15):    

    model.train()        

    for ii, (data, target) in enumerate(train_loader):        

        data, target = data.cuda(), target.cuda()              

        optimizer.zero_grad()

        output = model(data)                    

        loss = loss_func(output, target)

        loss.backward()

        optimizer.step()          

        if ii % 1000 == 0:

            loss_log.append(loss.item())       

    print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))
submit =test_df

test_data = ImageData(df = submit, data_dir = test_dir, transform = data_transf)

test_loader = DataLoader(dataset = test_data, shuffle=False)
submit["breed"]=0
submit.head()
%%time

# Prediction

predict = []

model.eval()

for i, (data, _) in enumerate(test_loader):

    data = data.cuda()

    output = model(data)  

    output = torch.nn.functional.softmax(output, dim=1)

    output = output.cpu().detach().numpy()    

    predict.append(output[0])
predict = np.array(predict)
predict.shape
predict = pd.DataFrame(predict)
predict.head()
predict.columns
predict.columns=lab_encoder.inverse_transform(predict.columns)
predict.head()
submit = pd.concat([submit["id"],predict],axis=1)
submit.head()
submit.to_csv('submission.csv', index=False)