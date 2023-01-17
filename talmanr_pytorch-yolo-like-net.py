# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# step 0: Laod packeges

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

from pathlib import Path



import torch

import torch.nn.functional as F

from torch import nn

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from torch.utils import data

import torchvision

import torchvision.transforms as transforms

from torch.autograd import Variable

input_path = Path("../input/kuzushiji-recognition/")

import cv2



%matplotlib inline

import os











# Any results you write to the current directory are saved as output.

os.listdir(input_path)

#os.listdir("../input/kuzushiji-recognition/train_images/")

# Step 1:Explore images shape  (I skip this stage by defoult)



MinMaxCalc = 0 

if MinMaxCalc :



    MinMax = np.zeros((2,2))

    MinMax[1,:] = 10000

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))



            if os.path.join(dirname, filename)[-1] == 'g':

                im = Image.open(os.path.join(dirname, filename))

                #print(np.array(im).shape)



                if np.array(im).shape[0] > MinMax[0,0] : 

                    MinMax[0,0] =   np.array(im).shape[0] 

                if np.array(im).shape[1] > MinMax[0,1] : 

                    MinMax[0,1] = np.array(im).shape[1] 

                if np.array(im).shape[0] < MinMax[1,0] : 

                    MinMax[1,0] = np.array(im).shape[0] 

                if np.array(im).shape[1] < MinMax[1,1] : 

                    MinMax[1,1] = np.array(im).shape[1] 

                    

else:

    MinMax = np.array([[5286., 3442.],

       [2353., 1750.]])

print(MinMax)
#Step 2: train-validetion split



df_train = pd.read_csv( input_path / 'train.csv')

unicode_trans = pd.read_csv( input_path / 'unicode_translation.csv')

train_image_path = input_path / "train_images"

test_image_path = input_path / "test_images"

unicode_map = {codepoint: char for codepoint, char in unicode_trans.values}
# import scipy.ndimage



# if 0 :

#     df_train["Height"] = ""

#     df_train["Width"] = ""

#     for i,im in enumerate(df_train["image_id"]):

#         height, width, channels = scipy.ndimage.imread("../input/kuzushiji-recognition/train_images/"+im+".jpg").shape

#         df_train["Height"][i] = height

#         df_train["Width"][i] = width

# read some labels exmple:

length = 5

split_labels = df_train["labels"][0].split()

for idx in range(len(split_labels) // length):

    start_idx = idx * length

    print(split_labels[start_idx : start_idx + length])

    if idx == 14:

        break
# split_labels = df_train["labels"][0].split()

# del split_labels[::5]

# split data frame

L_im = len(df_train)

RandOrd = np.array(range(L_im))



np.random.shuffle(RandOrd)



SplitFrac = 0.2

SplitInd = int(L_im*SplitFrac)

Train_Im = df_train.iloc[RandOrd[:SplitInd]]

Val_Im = df_train.iloc[RandOrd[SplitInd:]]
# Step 3:Define data-set for GPU training . 

# image is resize to 1024 pixels width and hight

class Dataset(data.Dataset):

    def __init__(self,DataPath, list_IDs,ListLabels,BatchSize,transforms,TestFlag=0):

        'Initialization'

        self.BatchSize = BatchSize

        self.list_IDs = list_IDs

        self.DataPath = DataPath

        self.ListLabels = ListLabels

        self.transforms = transforms

        if TestFlag ==0 :

            self.Fpath =  "train_images"

        else:

            self.Fpath =  "test_images"



    def __len__(self):

        'Denotes the total number of samples'

        return len(self.list_IDs)



    def __getitem__(self, index):

        'Generates one sample of data'

        # Select sample

        ID = self.list_IDs[index]

        if len(self.ListLabels)>1 :

            Label = self.ListLabels[index]

        else:

            Label = 0 

        

        

        # Load data and get label

        img = Image.open(self.DataPath +"/"+ID+".jpg")

        width, height = img.size

        #img = img.resize((256,256),resample=Image.BILINEAR)

        img = img.resize((1024,1024),resample=Image.BICUBIC)

        Torch_img = self.transforms(img)

        

        LabelOut = torch.zeros(2000)

        try:          

            split_labels = Label.split()

            del split_labels[::5]

            Label = torch.tensor(np.array(split_labels,dtype =np.float32 ),dtype=torch.float32)

            LabelOut[:len(Label)] = Label 

        except:

            Label = torch.tensor(0,dtype=torch.float32)

        return Torch_img, LabelOut ,ID,width, height
# Imagge transformations:

train_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



Val_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# Dataset load

batch_size = 6

DataPath = "../input/kuzushiji-recognition/train_images/"

training_set = Dataset(DataPath,list(Train_Im["image_id"]),list(Train_Im["labels"]),batch_size,train_transforms)

Val_set = Dataset(DataPath,list(Val_Im["image_id"]),list(Train_Im["labels"]),batch_size,train_transforms)

Test_Set = Dataset("../input/kuzushiji-recognition/test_images/",os.listdir("../input/kuzushiji-recognition/test_images/"),[],batch_size,Val_transforms,TestFlag=1) 



train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, num_workers=1,shuffle=True) # num_workers=12

valid_loader = torch.utils.data.DataLoader(Val_set, batch_size=batch_size, num_workers=1,shuffle=True)

test_loader = torch.utils.data.DataLoader(Test_Set, batch_size=batch_size, num_workers=1,shuffle=False)
# Step 4:Design Yolo-shape net

class YoloishNet(nn.Module):

    def __init__(self):

        super(YoloishNet,self).__init__()

        self.Fc_features = 128

        self.C1 = nn.Conv2d(3,8,3,padding=1)

        self.C2 = nn.Conv2d(8,16,3,padding=1)

        self.C3 = nn.Conv2d(16,32,3,padding=1)

        self.C4 = nn.Conv2d(32,64,3,padding=1)

        self.C5 = nn.Conv2d(64,128,3,padding=1)

        self.C6 = nn.Conv2d(128,128,3,padding=1)

        self.C7 = nn.Conv2d(128,10,3,padding=1)

    

        self.L_Relu = nn.LeakyReLU(0.1, inplace=True)

        self.UpConv1 = nn.ConvTranspose2d(256,256,4,stride=2)

        

        self.BN1 = nn.BatchNorm2d(8)

        self.BN2 = nn.BatchNorm2d(16)

        self.BN3 = nn.BatchNorm2d(32)

        self.BN4 = nn.BatchNorm2d(64)

        self.BN5 = nn.BatchNorm2d(128)

        self.BN6 = nn.BatchNorm2d(128)

        self.maxpoll = nn.MaxPool2d(2,2)

        self.maxpool2 = nn.MaxPool2d((2,1),(2,1))

        self.FC = nn.Linear(128*32*32,10*32*32)

        

        self.fc1 = nn.Linear(self.Fc_features,128)

        #self.fc2 = nn.Linear(128,self.NumClasses )

        self.dropout = nn.Dropout(0.45)

        self.Bat1 = nn.BatchNorm1d(self.Fc_features)

        

        

        

    def forward(self,x):

        # add sequence of convolutional and max pooling layers

        x = self.maxpoll(self.L_Relu(self.BN1(self.C1(x))))

        x = self.maxpoll(self.L_Relu(self.BN2(self.C2(x))))

        x = self.maxpoll(self.L_Relu(self.BN3(self.C3(x))))

        x = self.maxpoll(self.L_Relu(self.BN4(self.C4(x))))

        x = self.dropout(self.maxpoll(self.L_Relu(self.BN5(self.C5(x)))))

        x = self.C7(x)

        # flatten image input

        #print(x.shape)

       # x = x.view(-1, self.Fc_features)

        # add dropout layer

        #x = self.Bat1(x)

        # add 1st hidden layer, with relu activation function

        #x = self.dropout(F.relu(self.fc1(x)))

        # add dropout layer

        # add 2nd hidden layer, with relu activation function

        #x = torch.sigmoid(self.fc2(x))

        #x = self.fc2(x)

        return x
# define model and optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YoloishNet()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



model.to(device)
# Step 5:Plot images 

def PlotImageWithFrame(Im,labels,NumPic,h,w,TrueLabel = 1):

    

    

    

    Im = inputs[NumPic,:,:,:].cpu().detach().numpy()

    Im = np.moveaxis(Im,[0,1,2],[2,0,1])



    Im = Im - np.min(Im)

    Im = Im / np.max(Im) * 255

    Im = Im.astype(np.uint8)





    img = np.zeros((1024,1024,3), np.uint8)

    img[:,:,:] = Im[:,:,:]

    #plt.figure(figsize=(20,20))





    Lab = labels[NumPic]

    height1  =h[NumPic].cpu().detach().numpy()

    heightRetio = 1024/height1

    Width1 = w[NumPic].cpu().detach().numpy()

    WidthRetio = 1024/Width1 





    Lab = Lab[:np.min(np.where(Lab==0))].cpu().detach().numpy()



    for i in range(0,len(Lab),4):

        cv2.rectangle(img, (int(Lab[i]*heightRetio), int(Lab[i+1]*WidthRetio)), ( int((Lab[i]+Lab[i+2])*heightRetio), int((Lab[i+1]+Lab[i+3])*WidthRetio)), (0,100,0), 2) 

        #cv2.putText(img, 'fdsvfrewsf', (int(Lab[i]*heightRetio), int(Lab[i+1]*WidthRetio) - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,100,0), 2)

        print(int(Lab[i]*heightRetio),int(Lab[i+1]*WidthRetio),int((Lab[i+2])*heightRetio),int((Lab[i+3])*WidthRetio))

    plt.figure(figsize=(20,20))

    plt.imshow(img)
# Step 6: Yolo label and loss calculation

# takes a 4*n (n- number of BB) label and converte it to 10*32*32 labels

def CreatLabel(Lab_all,h_all,w_all):

    

    LabelesFullOut = np.zeros((len(h_all),10,32,32))



    for j in range(len(h_all)):

        

        h = h_all[j]

        w = w_all[j]

        Lab = Lab_all[j]

        

        

        Lab = Lab.detach().numpy()

        height1 = h.detach().numpy()

        heightRetio = 1024/height1

        Width1 = w.detach().numpy()

        WidthRetio = 1024/Width1 



        NumSubSquere  = 32 

        H_Center_Norm =  (Lab[::4]+Lab[2::4]/2)*heightRetio/NumSubSquere

        W_Center_Norm = (Lab[1::4]+Lab[3::4]/2)*WidthRetio/NumSubSquere

        H_box = Lab[2::4]

        W_box = Lab[3::4]

        LabelesFull = np.zeros((10,32,32))





        for i in range(len(H_Center_Norm)):

            H1 = int(np.floor(H_Center_Norm[i]))

            W1 = int(np.floor(W_Center_Norm[i]))





            Ind = 0 

            if LabelesFull[0,H1,W1] == 1:

                Ind += 5

            LabelesFull[Ind,H1,W1] = 1

            LabelesFull[Ind+1,H1,W1] = np.remainder(H_Center_Norm[i],1)*NumSubSquere

            LabelesFull[Ind+2,H1,W1] = np.remainder(W_Center_Norm[i],1)*NumSubSquere

            LabelesFull[Ind+3,H1,W1] = np.sqrt(H_box[i])

            LabelesFull[Ind+4,H1,W1] = np.sqrt(W_box[i])





        LabelesFullOut[j,:,:,:] = LabelesFull

    return torch.from_numpy(LabelesFullOut)





# Transform  back  labels

def TransformLabel(LabelesFull,NumIm):





    Label = []

    for i in range(32):

        for j in range(32):

            if LabelesFull[NumIm,0,i,j] == 1:

                Label.append(i+LabelesFull[NumIm,1,i,j]/32)

                print(i+LabelesFull[NumIm,1,i,j]/32)

                Label.append(j+LabelesFull[NumIm,2,i,j]/32)

                Label.append(np.square(LabelesFull[NumIm,3,i,j] ) ) 

                Label.append(np.square(LabelesFull[NumIm,4,i,j] ) )



                



    return np.array(Label)
# calculate loss based on YOLO paper (more or less)

import torch.nn.functional as F



def CalcMyLoss(LabelesFull,logps):

    LabelesFull = LabelesFull.float()

    #logps = logps.cpu()



    TotalFullLoss = 0

    for i in range(logps.shape[0]):



        #PredSum = torch.sum(torch.pow(LabelesFull[i,0,:,:]-torch.sigmoid(logps[i,0,:,:]),2))/(32**2)

#         PredSum = F.binary_cross_entropy(F.sigmoid(logps[i,0,:,:]), LabelesFull[i,0,:,:])

        PredSum = torch.sum(torch.pow( (logps[i,0,:,:])- LabelesFull[i,0,:,:],2))/(32**2)



        PredSum2 = torch.sum(torch.pow( (logps[i,5,:,:])- LabelesFull[i,5,:,:],2))/(32**2)

        

        NumBB = 0

        PositionLoss = 0

        HeightWidthLoss = 0 

        for m in range(32):

            for n in range(32):

                for k in [0,5] :

                    if LabelesFull[i,k,n,m] == 1:

                        PositionLoss += torch.pow( LabelesFull[i,k+1,n,m] - logps[i,k+1,n,m],2) + torch.pow( LabelesFull[i,k+2,n,m] - logps[i,k+2,n,m],2)    

                        HeightWidthLoss += torch.pow( LabelesFull[i,k+3,n,m] - logps[i,k+3,n,m],2) + torch.pow( LabelesFull[i,k+4,n,m] - logps[i,k+4,n,m],2)

                        NumBB += 1



        #print("Pred 1 loss: "+str(PredSum.detach().cpu() )+", Pred 2 loss: "+str(PredSum2.detach().cpu())+"Position Loss: "+str((PositionLoss/NumBB/32/6).detach().cpu())+", Height Width Loss: "+str((HeightWidthLoss/NumBB/32).detach().cpu()))

        TotalLoss = 10*PredSum + PredSum2*0.5 + PositionLoss/NumBB/32/6 + HeightWidthLoss/NumBB/32

        PredLossCoeff = 10

        PositionLoss = 0.2

        #TotalLoss = PredLossCoeff*PredSum  + PositionLoss/NumBB/32*PositionLoss + HeightWidthLoss/NumBB/32

        TotalLoss = PredLossCoeff*PredSum 

        TotalFullLoss += TotalLoss

        #print(TotalFullLoss)

        

        

    return TotalFullLoss

    

    

    
#def NonMaxSupression(logps):

# logps1 = logps.cpu().detach().numpy()

# print(type(logps1))

# for ImNum in range(logps1.shape[0]):

#     ImData = logps1[ImNum,:,:,:]

#     PredValues = ImData[0,:,:]

#     sortedI = np.argsort(PredValues,axis=None)

#     print(sortedI.shape)

    

# Step 7:Main NN loop

epochs = 6

valid_loss_min = np.Inf



import time



for epoch in range(epochs):

    start = time.time()

    

    #scheduler.step()

    model.train()

    

    train_loss = 0.0

    valid_loss = 0.0

    

    for inputs, labels,_,h,w in train_loader:

        

        LabelesFull = CreatLabel(labels,h,w)



        # Move input and label tensors to the default device

        inputs, LabelesFull = inputs.to(device), LabelesFull.to(device)

        optimizer.zero_grad()

        

        logps = model(inputs)

        

        loss = CalcMyLoss(LabelesFull,logps)

        print(loss)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
# Step 8: Predicted bounding box visualizetion.

def PlotImageWithFrame_2(Im,labelsFull,NumPic,TrueLabel = 1,Label2=[],BB_thr = 0.5):

    

    

    Im = inputs[NumPic,:,:,:].cpu().detach().numpy()

    Im = np.moveaxis(Im,[0,1,2],[2,0,1])



    Im = Im - np.min(Im)

    Im = Im / np.max(Im) * 255

    Im = Im.astype(np.uint8)





    img = np.zeros((1024,1024,3), np.uint8)

    img[:,:,:] = Im[:,:,:]

        #plt.figure(figsize=(20,20))





    Lab = labelsFull[NumPic,:,:,:]

    



    #     height1  =h[NumPic].cpu().detach().numpy()

    #     heightRetio = 1024/height1

    #     Width1 = w[NumPic].cpu().detach().numpy()

    #     WidthRetio = 1024/Width1 







    # for i in range(0,len(Lab),4):

    #     cv2.rectangle(img, (int(Lab[i]), int(Lab[i+1])), ( int((Lab[i]+Lab[i+2])), int((Lab[i+1]+Lab[i+3]))), (0,100,0), 2) 

    #     #cv2.putText(img, 'fdsvfrewsf', (int(Lab[i]*heightRetio), int(Lab[i+1]*WidthRetio) - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,100,0), 2)

    #     print(int(Lab[i]),int(Lab[i+1]),int((Lab[i+2])),int((Lab[i+3])))



    for i in range(32):

        for j in range(32):

            if Lab[0,i,j] == 1 :

                cv2.rectangle(img, (int(i*32+Lab[1,i,j]- (Lab[3,i,j]**2)/4), int(j*32+Lab[2,i,j]-(Lab[4,i,j]**2)/4)), (int(i*32+ (Lab[3,i,j]**2)/4+Lab[1,i,j] ),  int(j*32+(Lab[4,i,j]**2)/4) +Lab[2,i,j]), (0,100,0), 2) 

            if Lab[5,i,j] == 1 :

                cv2.rectangle(img, (int(i*32+Lab[5+1,i,j]- (Lab[5+3,i,j]**2)/4), int(j*32+Lab[5+2,i,j]-(Lab[5+4,i,j]**2)/4)), (int(i*32+ (Lab[5+3,i,j]**2)/4+Lab[5+1,i,j] ),  int(j*32+(Lab[5+4,i,j]**2)/4) +Lab[5+2,i,j]), (0,100,0), 2) 

            if TrueLabel == 0 and Lab[0,i,j] >BB_thr:

            #if TrueLabel == 0 and Label2[NumPic,0,i,j] ==1:

                cv2.rectangle(img, (int(i*32+Lab[1,i,j]- (Lab[3,i,j]**2)/4), int(j*32+Lab[2,i,j]-(Lab[4,i,j]**2)/4)), (int(i*32+ (Lab[3,i,j]**2)/4+Lab[1,i,j] ),  int(j*32+(Lab[4,i,j]**2)/4) +Lab[2,i,j]), (0,100,0), 2) 

            #if TrueLabel == 0 and  Label2[NumPic,0,i,j]==1 :#Lab[5,i,j] >0:

             #   cv2.rectangle(img, (int(i*32+Lab[5+1,i,j]- (Lab[5+3,i,j]**2)/4), int(j*32+Lab[5+2,i,j]-(Lab[5+4,i,j]**2)/4)), (int(i*32+ (Lab[5+3,i,j]**2)/4+Lab[5+1,i,j] ),  int(j*32+(Lab[5+4,i,j]**2)/4) +Lab[5+2,i,j]), (0,100,0), 2) 



    plt.figure(figsize=(20,20))

    plt.imshow(img)
a = 0

for inputs, labels,_,h,w in train_loader:

        

    LabelesFull = CreatLabel(labels,h,w)



        # Move input and label tensors to the default device

    inputs, LabelesFull = inputs.to(device), LabelesFull.to(device)

    optimizer.zero_grad()

    logps = model(inputs)

      

    if a == 1 :    

        break

    a +=1

    
NumIm = 1

PlotImageWithFrame_2(inputs,logps,NumIm,TrueLabel = 0,Label2 =LabelesFull,BB_thr=0.1)





plt.hist(logps[0,0,:,:].cpu().detach().numpy())
plt.hist(logps[0,0,:,:].cpu().detach().numpy()[:])