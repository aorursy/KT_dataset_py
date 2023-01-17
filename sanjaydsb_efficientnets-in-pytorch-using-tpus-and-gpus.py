accelerator_type = 'GPU' 
if(accelerator_type == 'TPU'):    

    !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

    !python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev

    import torch_xla

    import torch_xla.core.xla_model as xm

    device = xm.xla_device(n=4, devkind='TPU')

    import torch
import numpy as np

import pandas as pd

import os

import glob

import matplotlib.pyplot as plt

from PIL import Image

import matplotlib.image as mpimg

import cv2

import seaborn as sn

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder



import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator



from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint



!pip install ../input/efnwheelpy/efficientnet_pytorch-0.7.0-py3-none-any.whl

import efficientnet_pytorch

from efficientnet_pytorch import EfficientNet

import torch

import torchvision

from torch import Tensor

from torchvision import transforms

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import os
directory = '../input/landmark-recognition-2020/'

train_dir = '../input/landmark-recognition-2020/train/*/*/*/*'

test_dir = '../input/landmark-recognition-2020/test/*/*/*/*'

output_dir ='../output/kaggle/working/'

image_dir_train='../input/landmark-recognition-2020/train/'

image_dir_test='../input/landmark-recognition-2020/test/'

os.listdir(directory)
test = pd.read_csv(os.path.join(directory,'sample_submission.csv'))

test['image_']=test.id.str[0]+"/"+test.id.str[1]+"/"+test.id.str[2]+"/"+test.id+".jpg"

test.head()
train = pd.read_csv(os.path.join(directory,'train.csv'))

train["image_"] = train.id.str[0]+"/"+train.id.str[1]+"/"+train.id.str[2]+"/"+train.id+".jpg"

train["target_"] = train.landmark_id.astype(str)

train.head()
class createDataset(Dataset):

    def __init__(self, transform, image_dir, df, train_type = True):        

        self.df = df 

        self.image_dir = image_dir    

        self.transform = transform

        self.train_type=train_type

        

    def __len__(self):

        return self.df.shape[0] 

    

    def __getitem__(self,idx):

        image_id = self.df.iloc[idx].id

        image_name = f"{self.image_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"

        self.image = Image.open(image_name)              

        self.image = self.transform(self.image)

#         self.Y = torch.Tensor([self.df.iloc[idx].landmark_id]).type(torch.LongTensor)        

        if(self.train_type):

            return {'image':self.image, 

                    'label':self.df.iloc[idx].landmark_id}

        else:

            return {'image':self.image}
Threshold_count = 142



valid_landmark_df = pd.DataFrame(train['landmark_id'].value_counts()).reset_index()

valid_landmark_df.columns =  ['landmark_id', 'count_']

list_valid_landmarks = list(valid_landmark_df[valid_landmark_df.count_ >= Threshold_count]['landmark_id'].unique())

num_classes = len(list_valid_landmarks)

print("Total classes in training :", num_classes)

print(train.shape)

train= train[train.landmark_id.isin(list_valid_landmarks)]



valid_img = lambda img: os.path.exists(f'{image_dir_test}/{img[0]}/{img[1]}/{img[2]}/{img}.jpg')

if test.id.apply(valid_img).sum()==test.shape[0]:

    print('All Testing Images are valid')

else:

    print('Found invalid test Images')

    test=test.loc[test.id.apply(exists)]
# from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

label_encoder.fit(train.landmark_id.values)

print('found classes', len(label_encoder.classes_))

assert len(label_encoder.classes_) == num_classes



train.landmark_id = label_encoder.transform(train.landmark_id)
TRAIN_BS = 64

TEST_BS = 64

mean = (0.485, 0.456, 0.406)

std =  (0.229,0.225,0.224)

IMG_SIZE = 128

transformations = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE),interpolation=Image.NEAREST),

                                      transforms.ToTensor(),

                                      transforms.Normalize(mean,std)

                                     ]

                                    )    

train_data = createDataset(transform = transformations , df = train , image_dir = image_dir_train , train_type = True )

train_loader = DataLoader(dataset = train_data, 

                          batch_size = TRAIN_BS,

  #                      , num_workers =4

                          shuffle = False)



test_data = createDataset(transform = transformations , df = test , 

                          image_dir = image_dir_test ,

                          train_type = False )

test_loader = DataLoader(dataset = test_data, 

                         batch_size = TEST_BS

#                          , num_workers =4

                        )
class EfficientNet(nn.Module):

    def __init__(self, num_classes):

        super(EfficientNet, self).__init__()

        self.base = efficientnet_pytorch.EfficientNet.from_name(f'efficientnet-b0')

        self.base.load_state_dict(torch.load('../input/modelfnb0/efficientnet-b0-355c32eb.pth'))

        self.avg_pool = nn.AvgPool2d(3, stride=2)

        self.dropout = nn.Dropout(p=0.2)

#         self.batchnorm = nn.BatchNorm2d(100, affine=False)

        self.output_filter = self.base._fc.in_features        

        self.classifier = nn.Linear(self.output_filter, num_classes)

    def forward(self, x):

        x = self.base.extract_features(x)

        x = self.avg_pool(x).squeeze(-1).squeeze(-1)

        x = self.dropout(x)

#         x = self.batchnorm(x)

        x = self.classifier(x)

        return x
model = EfficientNet(num_classes=num_classes)

if(accelerator_type == 'TPU'): 

    model = model.to(device)

else:

    model = model.cuda()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-3, weight_decay=1e-4)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=1e-6)
!cp ../input/gappython/global_average_precision.py ./

import global_average_precision

y_true = {

        'id_001': 123,

        'id_002': None,

        'id_003': 999,

        'id_004': 123,

        'id_005': 999,

        'id_006': 888,

        'id_007': 666,

        'id_008': 666,

        'id_009': None,

        'id_010': 666,

    }

y_pred = {

        'id_001': (123, 0.90),

        'id_002': (123, 0.10),

        'id_003': (999, 0.90),

        'id_005': (999, 0.40),

        'id_007': (555, 0.60),

        'id_008': (666, 0.70),

        'id_010': (666, 0.99),

    }



def GAP( y_true,y_pred):

    return global_average_precision.global_average_precision_score(y_true, y_pred)



GAP(y_true, y_pred)
n_epochs = 1

loss_list = []

activation = nn.Softmax(dim=1)

loss_list=[]

gap_score_list=[]

model.train()

for epochs in range(n_epochs):    

    for i, data_x_y in enumerate(tqdm(train_loader)):        

        x= data_x_y['image']

        y=data_x_y['label']  

        

        optimizer.zero_grad()

        

        if(accelerator_type == 'TPU'): 

            yhat =  model(x.to(device))

            loss = criterion(yhat, y.to(device))   

            

        else:

            yhat =  model(x.cuda())

            loss = criterion(yhat, y.cuda())



        conf_scores, pred_labels = torch.max(yhat.detach(), dim=1)

        

        loss.backward()

        optimizer.step()

        lr_scheduler.step()

            

        tuple_pred = list(zip(pred_labels.cpu().numpy(),conf_scores.cpu().numpy()))

        true_labels =y.numpy()

        y_true = {}

        y_pred = {}



        for j in range(len(tuple_pred)):

            y_true[f'{j}'] = true_labels[j]

            y_pred[f'{j}'] = tuple_pred[j]



        gapscore = GAP(y_true, y_pred)

        gap_score_list.append(gapscore)

        loss_list.append(loss.detach())   

#         if(i==5):

#             break



        print(f" {i} LOSS <{loss}> GAP <{gapscore}> ") 

#         print(f"GAP {gapscore}") 
model.eval()



activation = nn.Softmax(dim=1)

all_predicts, all_confs = [], []



with torch.no_grad():    

    for i, data in enumerate(tqdm(test_loader)):

        input_ = data['image']



        yhat = model(input_.cuda())

        yhat = activation(yhat)



        confs, predicts = torch.topk(yhat, 1)

        all_confs.append(confs)

        all_predicts.append(predicts)

    predicts = torch.cat(all_predicts)

    confs = torch.cat(all_confs)



predicts_gpu, confs_gpu = predicts, confs

predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()

labels = [label_encoder.inverse_transform(pred) for pred in predicts]

labels = [l[0] for l in labels]

confidence_score = [score[0] for score in confs]
all_predicts[161].size()
for i in range(len(test)):

    test.loc[i, "landmarks"] = str(labels[i]) + " " + str(confidence_score[i])

        

del test['image_']

test.head()   
test.to_csv("submission.csv", index=False,float_format='%.6f')

test.head(20)
test