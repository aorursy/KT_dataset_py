import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import numpy as np

import os

import pandas as pd

import plotly.offline as py

import plotly.graph_objs as go

import seaborn as sns



plt.rcParams['figure.figsize']=(20,10)

print(os.listdir("../input"))

py.init_notebook_mode(connected=False)



%env JOBLIB_TEMP_FOLDER=/tmp
df = pd.read_csv('../input/labels.csv')
df.head()
df['breed'].describe()
temp = pd.DataFrame({'breed': df['breed'].value_counts().index, 'instances': df['breed'].value_counts().values})

temp = temp.sort_values(by=['breed'])

temp.head()
trace = go.Bar(x=temp['breed'], y=temp['instances'])

data = [trace]

layout = go.Layout(

        title='Breed Counts',

        autosize=False,

        width=5000,

        height=500,

        margin=dict(

            l=100,

            r=100,

            b=100,

            t=100

        )

    )

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
df['breed'] = pd.get_dummies(df['breed']).values.tolist()
df.head()
from torchvision import transforms

from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset

from PIL import Image



import torchvision.models as models

import torch.nn as nn

import torch.optim as optim
df['image_path'] = '../input/train/' + df['id'].astype(str) + '.jpg'
df.head()
labels = torch.tensor(df['breed'].tolist())
class CustomDataset(Dataset):

    def __init__(self, image_path, labels, train=True):

        self.image_path = image_path

        self.labels = labels

        self.transform = transforms.Compose([

                        transforms.Resize(255),

                        transforms.CenterCrop(224),

                        transforms.ToTensor(),

                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                    ])



    def __getitem__(self, index):

        image = Image.open(self.image_path[index])

        t_image = self.transform(image)

        

        return t_image, self.labels[index]



    def __len__(self):  # return count of sample we have

        return len(self.image_path)
train_dataset = CustomDataset(df['image_path'], labels, train=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)

model = models.densenet161(pretrained=True)

# Turn off training for their parameters

for param in model.parameters():

    param.requires_grad = False
classifier = nn.Sequential(nn.Linear(255, 1024),

                           nn.ReLU(),

                           nn.Linear(1024, 512),

                           nn.ReLU(),

                           nn.Linear(512, 120),

                           nn.LogSoftmax(dim=1))

# Replace default classifier with new classifier

model.classifier = classifier
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device specified above

model.to(device)

criterion = nn.NLLLoss()

# Set the optimizer function using torch.optim as optim library

optimizer = optim.Adam(model.classifier.parameters())