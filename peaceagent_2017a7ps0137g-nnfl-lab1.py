# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import torch

from torch import nn

import torch.nn.functional as F 

from torch.utils.data import Dataset, DataLoader

import torchvision

import torchvision.transforms as transforms

from skimage import io, transform

import time

seed_val = 1

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)

from collections import Counter

import warnings

warnings.filterwarnings('ignore')
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

print("using ", device)
from IPython.display import HTML

import base64



def create_download_link(df, title = "Download CSV file",i= [0]):  

    csv = df.to_csv(index=False)

    i[0] = i[0]+1

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename='resnext'+str(i[0])+'.csv')

    return HTML(html)
images =list(os.listdir("/kaggle/input/nnfl-lab-1/training/training/"))

image_dict={i:i.split('.')[0].split('_')[0] for  i in images}

train_df = pd.DataFrame.from_dict({"image_id":image_dict.keys(),"label":image_dict.values()})
Counter(train_df["label"])
train_df, validate_df = train_test_split(train_df, test_size=0.10,stratify = train_df["label"])

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)

Counter(validate_df["label"])
train_df['label']=train_df['label'].apply(lambda x:{'chair':0,'kitchen':1,'knife':2,'saucepan':3}[x])

validate_df['label']=validate_df['label'].apply(lambda x:{'chair':0,'kitchen':1,'knife':2,'saucepan':3}[x])
class ImageDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):

        self.df = df

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.df['image_id'][idx])         

        image = Image.open(img_name).convert('RGB')                               

        label = torch.tensor(self.df['label'][idx])                         

        

        if self.transform:            

            image = self.transform(image)                                          

        

        sample = (image, label)        

        return sample
train_transform = transforms.Compose([transforms.Resize((224, 224)),

                                transforms.ColorJitter(),

                                transforms.RandomGrayscale(p=0.1),

                                transforms.RandomHorizontalFlip(p=0.5),

                                transforms.RandomRotation(degrees=20),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

eval_transform = transforms.Compose([transforms.Resize((224, 224)),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
batch_size=32

train_path = '/kaggle/input/nnfl-lab-1/training/training'

test_path = '/kaggle/input/nnfl-lab-1/testing/testing'



train_set = ImageDataset(train_df, train_path, train_transform)

val_set = ImageDataset(validate_df, train_path, eval_transform)



train_loader = DataLoader(train_set, batch_size=batch_size,

                        shuffle=True, num_workers=0)

val_loader = DataLoader(val_set, batch_size=400,

                        shuffle=False, num_workers=0)
model = torchvision.models.resnext50_32x4d(pretrained=False)

model.fc = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(2048, 512),nn.LeakyReLU(inplace=True),nn.Linear(512,4))
! conda install -y gdown
! gdown https://drive.google.com/uc?id=1RUXcOB0aVux9JKD-PCJ2xyYsya3wuTnb
model = torch.load('resnext.h5')

model = model.to(device)
!pip install torchsummary
from torchsummary import summary

summary(model,input_size=(3,224,224))
optimizer = torch.optim.Adam(model.parameters(),

                             lr=0.0001, 

                             eps=10e-8,

                             weight_decay=0)

criterion = nn.CrossEntropyLoss()
SAVE_PATH = './resnext.h5'

max_accuracy = -0.1
# model.train()

# epochs = 10

# for epoch in range(epochs):

#     running_loss = 0.0

#     epoch_loss = 0.0

#     for i, data in enumerate(train_loader):

#         images, labels = data

#         images, labels = images.to(device), labels.to(device)

        

#         optimizer.zero_grad()



#         logits = model(images)

#         loss = criterion(logits, labels.long())



#         loss.backward()

#         optimizer.step()

        

#         running_loss += loss.item()

#         epoch_loss += loss.item()

        

#         if i % 10 == 9:    # print every 10 mini-batches

#                 print('[%d, %5d] loss: %.3f ' %(epoch + 1, i + 1, running_loss / (10)))

#                 running_loss = 0.0





#     print("\nEPOCH ", epoch+1, " TRAIN LOSS = ", epoch_loss/len(train_loader))



#     val_loss = 0.0

#     model.eval()

#     preds = []

#     ground_truth = []

#     with torch.no_grad():

#         for i, data in enumerate(val_loader):

#             images, labels = data

#             images, labels = images.to(device), labels.to(device)

            

#             logits = model(images)

#             loss = criterion(logits, labels.long())

#             val_loss += loss.item()



#             _, predictions = torch.max(logits, 1)

#             for pred in predictions:

#                 preds.append(int(round(pred.item())))

#             for label in labels:

#                 ground_truth.append(int(label.item()))

#     preds = np.array(preds)

#     ground_truth = np.array(ground_truth)

#     accuracy = accuracy_score(ground_truth, preds)

#     print("EPOCH ", epoch+1, " VAL LOSS = ", val_loss/len(val_loader),"ACC:", accuracy, '\n')

#     model.train()

#     if max_accuracy < accuracy:

#         print("Model optimized, saving weights ...\n")

#         torch.save(model, SAVE_PATH)

#         min_val_loss = (val_loss/len(val_set))

#         max_accuracy = accuracy

# torch.save(model, 'resnext.h5')
test_df = pd.read_csv('/kaggle/input/nnfl-lab-1/sample_sub.csv')

test_df.columns = ["image_id","label"]

test_df['label'] = [0]*len(test_df)

test_set = ImageDataset(test_df, test_path, eval_transform)

test_loader = DataLoader(test_set, batch_size=200,

                        shuffle=False, num_workers=0)
model.eval()

preds = []

with torch.no_grad():

    for i, data in enumerate(test_loader):

        images, labels = data

        images, labels = images.to(device), labels.to(device)

            

        logits = model(images)



        _, predictions = torch.max(logits, 1)

        for pred in predictions:

            preds.append(int(round(pred.item())))
test_df['label'] = preds
test_df.columns = ["id","label"]

test_df.to_csv('resnext.csv', index=False)

create_download_link(test_df)