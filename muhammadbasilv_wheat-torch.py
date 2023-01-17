# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

from torch import nn

import torch.nn.functional as F

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2
class Lambda(nn.Module):

    def __init__(self, func):

        super().__init__()

        self.func = func



    def forward(self, x):

        return self.func(x)





def preprocess(x):

    return x.view(-1, 3, 256, 256) 



def flatten(x):

  size=x.size()[1:]

  num_features=1

  for s in size:

    num_features*=s

  return x.view(-1,num_features)



def reshape(x):

  return x.view(16,16,5)
yolo_model=nn.Sequential(

    Lambda(preprocess),

    #layer 1

    nn.Conv2d(3,32,3,padding=1),

    nn.BatchNorm2d(32),

    nn.LeakyReLU(negative_slope=0.1),

    nn.MaxPool2d(2),

    #layer 2

    nn.Conv2d(32,32,3,padding=1),

    nn.BatchNorm2d(32),

    nn.LeakyReLU(negative_slope=0.1),

    #layer 3

    nn.Conv2d(32,64,3,padding=1),

    nn.BatchNorm2d(64),

    nn.LeakyReLU(negative_slope=0.1),

    nn.MaxPool2d(2),

    #layer 4

    nn.Conv2d(64,128,3,padding=1),

    nn.BatchNorm2d(128),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 5

    nn.Conv2d(128,128,1,padding=0),

    nn.BatchNorm2d(128),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 6

    nn.Conv2d(128,256,3,padding=1),

    nn.BatchNorm2d(256),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 7

    nn.Conv2d(256,256,1,padding=0),

    nn.BatchNorm2d(256),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 8

    nn.Conv2d(256,512,3,padding=1),

    nn.BatchNorm2d(512),

    nn.LeakyReLU(negative_slope=0.1),

    nn.MaxPool2d(2),



    #layer 9

    nn.Conv2d(512,256,1,padding=0),

    nn.BatchNorm2d(256),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 10

    nn.Conv2d(256,512,3,padding=1),

    nn.BatchNorm2d(512),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 11

    nn.Conv2d(512,256,1,padding=0),

    nn.BatchNorm2d(256),

    nn.LeakyReLU(negative_slope=0.1),

    

    #layer 12

    nn.Conv2d(256,512,3,padding=1),

    nn.BatchNorm2d(512),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 13

    nn.Conv2d(512,256,1,padding=0),

    nn.BatchNorm2d(256),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 14

    nn.Conv2d(256,512,3,padding=1),

    nn.BatchNorm2d(512),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 15

    nn.Conv2d(512,256,1,padding=0),

    nn.BatchNorm2d(256),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 16

    nn.Conv2d(256,512,3,padding=1),

    nn.BatchNorm2d(512),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 17

    nn.Conv2d(512,512,1,padding=0),

    nn.BatchNorm2d(512),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 18

    nn.Conv2d(512,1024,3,padding=1),

    nn.BatchNorm2d(1024),

    nn.LeakyReLU(negative_slope=0.1),

    nn.MaxPool2d(2),



    #layer 19

    nn.Conv2d(1024,512,1,padding=0),

    nn.BatchNorm2d(512),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 20

    nn.Conv2d(512,1024,3,padding=1),

    nn.BatchNorm2d(1024),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 21

    nn.Conv2d(1024,512,1,padding=0),

    nn.BatchNorm2d(512),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 22

    nn.Conv2d(512,1024,3,padding=1),

    nn.BatchNorm2d(1024),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 23



    nn.Conv2d(1024,1024,3,padding=1),

    nn.BatchNorm2d(1024),

    nn.LeakyReLU(negative_slope=0.1),

    nn.MaxPool2d(2),



    #layer 24

    nn.Conv2d(1024,1024,3,padding=1),

    nn.BatchNorm2d(1024),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 25

    nn.Conv2d(1024,1024,3,padding=1),

    nn.BatchNorm2d(1024),

    nn.LeakyReLU(negative_slope=0.1),



    #layer 26

    nn.Conv2d(1024,128,3,padding=1),

    nn.BatchNorm2d(128),

    nn.LeakyReLU(negative_slope=0.1),





    Lambda(flatten),



    #fc

    nn.Linear(8*8*128,1280),



    Lambda(reshape)





)
def yolo_loss(y_true,y_pred):

  lossx=torch.sum(torch.square((y_true[:,:,0:1]-y_pred[:,:,0:1]))*y_true[:,:,4:])

  lossy=torch.sum(torch.square((y_true[:,:,1:2]-y_pred[:,:,1:2]))*y_true[:,:,4:])

  loss1=torch.add(lossx,lossy)

  lossw=torch.sum(torch.square((y_true[:,:,2:3]-y_pred[:,:,2:3]))*y_true[:,:,4:])

  lossh=torch.sum((torch.square((y_true[:,:,3:4])-(y_pred[:,:,3:4])))*y_true[:,:,4:])

  loss2=torch.add(lossw,lossh)

  loss_xy_wh=torch.add(loss1,loss2)

  lossC=torch.sum(torch.square((y_true[:,:,4:]-y_pred[:,:,4:]))*y_true[:,:,4:])

  lossC+=torch.sum(torch.square((y_true[:,:,4:]-y_pred[:,:,4:]))*(1-y_true[:,:,4:]))/16  

  

    

  total_loss=torch.add(loss_xy_wh,lossC)

  return total_loss
data=pd.read_csv('../input/global-wheat-detection/train.csv')

data.head()
def get_list(a):

  a=list(a.strip('[').strip(']'))

  val='0'

  liz=[]

  for i in a:

    if(i!=',' and i!=' '):

      val+=i

    elif (i==' '):

      liz.append(float(val))

      val='0'  

  liz.append(float(val))

  return liz



ids=[]

bboxes=[]



for index,id in enumerate(data['image_id']):

  count=0

  for i in ids:

    if(i==id):

      count+=1

  if(count==0):

    bbox_array=np.zeros((16,16,5))

    for index2,id2 in enumerate(data['image_id']):

      if(id2==id):

        bboz=get_list(data['bbox'][index2])

        w=int(bboz[2])

        h=int(bboz[3])

        x=int(bboz[0]+bboz[2]/2)

        y=int(bboz[1]+bboz[3]/2)

        box1=int(x/64)

        box2=int(y/64)

        bbox_array[box1,box2,:]=(x%64)/64,(y%64)/64,w/1024,h/1024,1

    bboxes.append(bbox_array)

    ids.append(id)

  print(index)
device = torch.device("cuda:0")

model=yolo_model.to(device)
torch.cuda.is_available()
bbox_tensor=torch.FloatTensor(bboxes)
opt=torch.optim.SGD(yolo_model.parameters(),lr=0.0001,momentum=0.9)
train_data=[]

#train_data=np.zeros((1,512,512,3))

for index,id in enumerate(ids):

    filename=id

    filename='../input/global-wheat-detection/train/'+id+'.jpg'

    img=cv2.imread(filename)

    img=cv2.resize(img,(256,256))

    img=img/255.0

    #img=np.reshape(img,(1,256,256,3))

    train_data.append(img)

    print(index)


for epochs in range(150):

  for i in range(3000):

    ix=int(np.random.randint(0,3335,1))

    pred=yolo_model(torch.FloatTensor(train_data[ix]).to(device))

    loss=yolo_loss(bbox_tensor[ix].to(device),pred)

    print(loss,i,epochs)

    loss.backward()

    opt.step()

    opt.zero_grad()

    

    
nam = '../input/global-wheat-detection/train/'+ids[1001]+'.jpg'

img=cv2.imread(nam)

img=cv2.resize(img,(256,256))

test_img=img/255.0

test=yolo_model(torch.FloatTensor(test_img).to(device))

no=0

for i in range(16):

    for j in range(16):

        if(test[i][j][4]>=0.9):

            x=(i*64)+(test[i][j][0]*64)

            y=(j*64)+(test[i][j][1]*64)

            w=test[i][j][2]*1024

            h=test[i][j][3]*1024

            x1=(x-w/2)/4

            y1=(y-h/2)/4

            x2=(x+w/2)/4

            y2=(y+h/2)/4

            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)

            no+=1

plt.imshow(img[:,:,[2,1,0]])  

fig=plt.gcf()

fig.set_size_inches((12,12))



print(no)
test_list=['../input/global-wheat-detection/test/2fd875eaa.jpg','../input/global-wheat-detection/test/348a992bb.jpg','../input/global-wheat-detection/test/51b3e36ab.jpg','../input/global-wheat-detection/test/51f1be19e.jpg','../input/global-wheat-detection/test/53f253011.jpg','../input/global-wheat-detection/test/796707dd7.jpg','../input/global-wheat-detection/test/aac893a91.jpg','../input/global-wheat-detection/test/cb8d261a3.jpg','../input/global-wheat-detection/test/cc3532ff6.jpg','../input/global-wheat-detection/test/f5a1f0358.jpg']

test_imgs=[]

test_preds=[]

for i in test_list:

    img=cv2.imread(i)

    img=cv2.resize(img,(256,256))

    test_img=img/255.0

    test=yolo_model(torch.FloatTensor(test_img).to(device))

    test_preds.append(test)

    no=0

    for i in range(16):

        for j in range(16):

            if(test[i][j][4]>=0.9):

                x=(i*64)+(test[i][j][0]*64)

                y=(j*64)+(test[i][j][1]*64)

                w=test[i][j][2]*1024

                h=test[i][j][3]*1024

                x1=(x-w/2)/4

                y1=(y-h/2)/4

                x2=(x+w/2)/4

                y2=(y+h/2)/4

                cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)

                no+=1

    test_imgs.append(img)            

    plt.imshow(img[:,:,[2,1,0]])  

    fig=plt.gcf()

    fig.set_size_inches((12,12))
len(test_preds)