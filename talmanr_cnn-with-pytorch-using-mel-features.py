

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import numpy as np 

import pandas as pd

import os

import librosa

import librosa.display



import IPython.display as ipd



import matplotlib.pyplot as plt

print(os.listdir("../input"))







import torchaudio

import torch.nn as nn

import torch

import torch.nn.functional as F

from torch.utils import data

from torchvision import datasets, models, transforms

import torch.optim as optim



train_on_gpu=torch.cuda.is_available()



# Any results you write to the current directory are saved as output.
import zipfile



# Will unzip the files so that you can see them..

with zipfile.ZipFile("../input/"+"train_curated"+".zip","r") as z:

    z.extractall(".")

    

    





Labels = pd.read_csv("../input/train_curated.csv")

Labels.head()

WavPath = '/kaggle/working/'

Fils = os.listdir(WavPath)

sound, sample_rate = torchaudio.load(WavPath+Fils[2])

ipd.Audio(data=sound[0,:],rate=sample_rate) # load a local WAV file

x, sr = librosa.load(WavPath+Fils[2])



plt.figure(figsize=(14, 5))







librosa.display.waveplot(x, sr=sr)

X = librosa.stft(x)

Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))

#librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')

Xdb.shape



S = librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=128)

log_S = librosa.power_to_db(S, ref=np.max)

MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=23)

delta2_mfcc = librosa.feature.delta(MFCC, order=2)



#MFCC = librosa.feature.mfcc(y=x, sr=sample_rate,n_mfcc=23,dct_type=2)

librosa.display.specshow(log_S)

#print(np.max(MFCC),np.min(MFCC))

#MFCC = (MFCC+200)/500

#print(np.max(MFCC),np.min(MFCC))

plt.colorbar()

plt.tight_layout()
FilesS = np.zeros(len(Fils))

for i,File in enumerate(Fils):

    FilesS[i] = os.path.getsize(WavPath+File)



plt.figure(figsize=(20,8))

plt.hist(FilesS,bins=50)
Fils_2 = Labels['fname']

Fils_2



Class =set(Labels['labels'])

All_class= [] 

for i in Class:

    for j  in i.split(','):

        All_class.append(j)



All_class = set(All_class)



NumClasses = len(All_class)

OneHot_All = np.zeros((len(Fils_2),NumClasses))



for  i,file in enumerate(Labels['labels']):

    for j,clas in enumerate(All_class):

        OneHot_All[i,j] = np.int(clas in file)



np.mean(log_S/10+4)
# Encode classes

#ClassDict = dict(enumerate(set(Labels['labels'])))

#Class2int = {ch: ii for ii, ch in ClassDict.items()}

#encoded = np.array([Class2int[ch] for ch in Labels['labels']])



#NumClasses = len(Class2int) 

print(NumClasses)

## split data into training, validation, and test data (features and labels, x and y)

split_frac = 0.79

batch_size = 32



split_idx = int(len(Fils)*split_frac)

split_idx1 = int(batch_size*np.floor(split_idx/batch_size))

split_idx2 = int(batch_size*np.floor( (len(Fils) - split_idx1)/batch_size ))

train_x, val_x = Fils_2[:split_idx1], Fils_2[split_idx1:split_idx1+split_idx2]

train_y, val_y = OneHot_All[:split_idx1,:], OneHot_All[split_idx1:split_idx1+split_idx2,:]

print(len(train_x)/batch_size, len(val_x)/batch_size )



from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split( Fils_2, OneHot_All, test_size=1-split_frac, random_state=42)

print(train_x.shape,val_x.shape,train_y.shape,val_y.shape)





from scipy.io import wavfile

from librosa.feature import mfcc

class Dataset(data.Dataset):

    def __init__(self, list_IDs, labels,DataPath,RecLen,DecNum=5,fft_Samp= 256,Im_3D= False):

        'Initialization'

        self.labels = labels

        self.list_IDs = list_IDs

        self.DataPath = DataPath

        self.RecLen = RecLen # length of most records

        self.fft_Samp = fft_Samp 

        self.Im_3D = Im_3D

        

        self.NFCC_Num = 128

        self.TimeSamp = 128

    def __len__(self):

        'Denotes the total number of samples'

        return len(self.list_IDs)



    def __getitem__(self, index):

        'Generates one sample of data'

        ID = self.list_IDs[index]



        #y, sr = librosa.load(self.DataPath + ID)

        data,fs =  librosa.load(self.DataPath + ID)

        data = np.float32(data)

        S = librosa.feature.melspectrogram(data, sr=sample_rate, n_mels=128)

        Mel = librosa.power_to_db(S, ref=np.max)/10+4

        LabelOut = torch.from_numpy(self.labels[ID]).double()

        

        

        Im = torch.zeros((self.NFCC_Num,self.TimeSamp)).type(torch.FloatTensor)

        Ssum = np.sum(Mel,axis=0)

        MaxE = np.argmax(Ssum)

        if MaxE > Mel.shape[1]-64 : 

            MaxE = Mel.shape[1]-65

        if MaxE< 64 :

            MaxE = 64

        if Mel.shape[1] > self.TimeSamp :

            Im = torch.from_numpy(Mel[:,MaxE-64:MaxE+64])

        else: 

            Im[:,:Mel.shape[1]  ] = torch.from_numpy(Mel)

        

        



        Im = Im.double()

        return Im, LabelOut,ID
class CnnAudioNet(nn.Module):

    def __init__(self,NumClasses):

        super(CnnAudioNet,self).__init__()

        self.NumClasses = NumClasses

        self.Fc_features = 128

        self.C1 = nn.Conv2d(1,32,5,padding=1)

        self.C2 = nn.Conv2d(32,32,5,padding=1)

        self.C3 = nn.Conv2d(32,64,5,padding=1)

        self.C4 = nn.Conv2d(64,64,5,padding=1)

        

        self.BN1 = nn.BatchNorm2d(32)

        self.BN2 = nn.BatchNorm2d(64)

        self.BN3 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d(2,2)

        self.maxpool2 = nn.MaxPool2d((1,2),(1,2))

        

        

        self.fc1 = nn.Linear(64*8*8,128)

        self.fc2 = nn.Linear(128,self.NumClasses )

        self.dropout = nn.Dropout(0.25)

        self.Bat1 = nn.BatchNorm1d(128)



        

        

    def forward(self,x):

        # add sequence of convolutional and max pooling layers

        x = F.relu(self.BN1(self.C1(x)))

        x = self.maxpool1(F.relu(self.BN1(self.C2(x))))

        x = F.relu(self.BN2(self.C3(x)))

        x = self.maxpool1(F.relu(self.BN2(self.C4(x))))

        x = F.relu(self.BN2(self.C4(x)))

        x = self.maxpool1(F.relu(self.BN2(self.C4(x))))

        x = F.relu(self.BN2(self.C4(x)))

        x = F.relu(self.BN3(self.C4(x)))

        # flatten image input

        x = self.dropout(x.view(-1,64*8*8))

        # add dropout layer

        x =  self.dropout(self.fc1(x))

        # add 1st hidden layer, with relu activation function

        # add dropout layer

        # add 2nd hidden layer, with relu activation function

        #x = torch.sigmoid(self.fc2(x))

        x = self.fc2(x)

        return x

        
from torchvision import datasets, models, transforms





# Freeze training for all layers





class CnnTransferNet(nn.Module):

    def __init__(self):

        super(CnnTransferNet,self).__init__()

        

        self.vgg =  models.vgg16_bn().cuda()

        for param in self.vgg.features.parameters():

            param.require_grad = False



        

        self.fc1 = nn.Linear(1000,128)

        self.fc2 = nn.Linear(128,NumClasses)

        self.dropout = nn.Dropout(0.25)



        

        

    def forward(self,x):

        # add sequence of convolutional and max pooling layers

        Features = self.dropout(self.vgg(x))

        # flatten image input

        # add 1st hidden layer, with relu activation function

        Features = F.relu(self.fc1(Features))

        # add dropout layer

        # add 2nd hidden layer, with relu activation function

        Features = self.fc2(Features)

        return Features
model = CnnAudioNet(NumClasses)

if train_on_gpu:

    model.cuda()

print(model)

# specify loss function (MSE)



#criterion = nn.MSELoss()

#criterion = nn.BCELoss()

criterion = nn.BCEWithLogitsLoss()

#criterion = nn.MultiLabelSoftMarginLoss()



optimizer = optim.Adam(params=model.parameters(), lr=0.001)# specify optimizer

#optimizer = optim.Adam(model.parameters(), lr=0.005)





a = train_x.tolist()

#abelsDict = dict(zip(Fils,one_hot))

labelsDict_train = dict(zip(train_x,train_y))

labelsDict_val = dict(zip(val_x,val_y))



params = {'batch_size': batch_size,

          'shuffle': True,

          'num_workers': 9}

params_v = {'batch_size': batch_size,

          'shuffle': False,

          'num_workers': 3}

RecLen = 176400



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])



training_set = Dataset(train_x.tolist(), labelsDict_train,WavPath,RecLen,transforms.Compose(normalize))

training_generator = data.DataLoader(training_set, **params)



val_set = Dataset(val_x.tolist(),labelsDict_val,WavPath,RecLen,transforms.Compose(normalize))

val_generator = data.DataLoader(val_set, **params_v)





import time

start_time = time.time()

#Warnings.filterwarnings('ignore')



# number of epochs to train the model

n_epochs = 1



valid_loss_min = np.Inf # track change in validation loss

print("Start training:")

idx = 0 

for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    TotMSE = 0 

    TotEl = 0

    

    ###################

    # train the model #

    ###################

    model.train()



    for dataBatch, target,_ in training_generator:

        

        idx+=1



        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            dataBatch, target = dataBatch.unsqueeze(1).float().cuda(), target.cuda()

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(dataBatch)

        # calculate the batch loss

        #loss = criterion(output, torch.squeeze(torch.argmax(target,dim=-1)))

        loss = criterion(output,target.float())

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update training loss

        train_loss += loss.item()*dataBatch.size(0)

        #print(loss.item())

        #print('Finish batch')

        _,pred = torch.max(output,1)

        

        #Correct = torch.sum(torch.pow(output-target.float(),2))#

        ErrorS = torch.sum(torch.pow(torch.sigmoid(output)-target.float(),2))#

        TotMSE += ErrorS

        TotEl += output.numel()

        Correct =torch.sum(pred ==torch.squeeze(torch.argmax(target,dim=-1)))

        #print('Train batch loss: {:.6f},  Error: {:.4f},  Sum Correct: {} out of {}'.format(loss,ErrorS,Correct,output.shape[0]))

    print('Epoch: {} \t  Train batch loss: {:.6f} '.format(epoch,loss))



        

    ######################    

    # validate the model #

    ######################

    with torch.no_grad():

        model.eval()

        TotEl_v = 0

        valid_loss = 0 

        TotMSE_v = 0

        for dataBatch_v, target ,_ in val_generator  :



        # move tensors to GPU if CUDA is available

            if train_on_gpu:

                dataBatch_v, target = dataBatch_v.unsqueeze(1).float().cuda(),target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

            output = model(dataBatch_v)

        # calculate the batch loss

            loss = criterion(output,target.float())



            #loss = criterion(output, torch.squeeze(torch.argmax(target,dim=-1)))

        # update average validation loss 

            output.shape

            _,pred = torch.max(output,1)

            Correct = torch.sum(pred ==torch.squeeze(torch.argmax(target,dim=-1)))

            #SumCorrectVal += Correct

            valid_loss += loss.item()*dataBatch.size(0)

            #print(TotVal)



            ErrorS = torch.sum(torch.pow(torch.sigmoid(output)-target.float(),2))#

            TotMSE_v += ErrorS

            TotEl_v += output.numel()

        # calculate average losses

        train_lossM = train_loss/len(training_generator.dataset)

        valid_lossM = valid_loss/len(val_generator.dataset)

        MSE = TotMSE/TotEl

        MSE_V = TotMSE_v/TotEl_v



        # print training/validation statistics 

        print('Epoch: {} \t Training Loss: {:.6f}, Train MSE: {:.4f} \tValidation Loss: {:.6f},  Val MSE: {:.4f} '.format(

            epoch, train_lossM,MSE, valid_lossM,MSE_V))

        print("--- %s seconds ---" % (time.time() - start_time))
# data,target ,_= next(iter(val_generator))

# data = data.unsqueeze(1).float().cuda()

# output = model(dataBatch_v)

# plt.figure(figsize=(20,20))

# for i in range(16):

#     plt.subplot(4,4,i+1)

#     plt.plot(target[i,:].detach().cpu().numpy())

#     plt.plot(torch.sigmoid(output[i,:]).detach().cpu().numpy())

from glob import glob

F1 = glob('./*wav*')

for file in F1:

    os.remove(file)
# # Will unzip the files so that you can see them..

with zipfile.ZipFile("../input/"+"test"+".zip","r") as z:

    z.extractall("./test/")
WavPath_test =  './test/'



Fils_test = os.listdir(WavPath_test)





one_hot_test = np.zeros((len(Fils_test),NumClasses))



labelsDict = dict(zip(Fils_test,one_hot_test))



params = {'batch_size': 4,

          'shuffle': True,

          'num_workers': 4}



test_set = Dataset(Fils_test, labelsDict,WavPath_test,RecLen,transforms.Compose(normalize))

test_generator = data.DataLoader(test_set, **params)
model.eval()

SoftM = torch.nn.Softmax()

Output_all = [] 

BatchRecs_all = [] 

with torch.no_grad():                   # operations inside don't track history



    for dataBatch, Lab,BatchRecs in test_generator:

        if train_on_gpu:

            dataBatch, Lab = dataBatch.unsqueeze(1).float().cuda(), Lab.cuda()

        output = model(dataBatch)

        outP = torch.sigmoid(output)

        #outP = output

        Output_all.append(outP)

        BatchRecs_all.append(BatchRecs)



  
Dataout = np.zeros((4*len(Output_all)-3,80))

Names = []

for i in range(len(Output_all)):

    Dataout[i*4:(i+1)*4,:] = Output_all[i].cpu().detach().numpy()

    Names.append(BatchRecs_all[i][0])

    if i<840:

        Names.append(BatchRecs_all[i][1])

        Names.append(BatchRecs_all[i][2])

        Names.append(BatchRecs_all[i][3])

    

Cl = list(All_class)

#Cl.append('fname')



Output_all_DF =pd.DataFrame(columns=Cl,data = Dataout)

Output_all_DF['fname'] = Names

Output_all_DF.to_csv('submission.csv', index=False)

Output_all_DF.head()




