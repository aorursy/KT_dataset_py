import torch

import sys

import torch

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torchvision import transforms

from torch import nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable



from sklearn import cross_validation

from sklearn import metrics

from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc

from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split



print('__Python VERSION:', sys.version)

print('__pyTorch VERSION:', torch.__version__)



import numpy

import numpy as np



use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

Tensor = FloatTensor



import pandas

import pandas as pd



import logging

handler=logging.basicConfig(level=logging.INFO)

lgr = logging.getLogger(__name__)

%matplotlib inline



# !pip install psutil

import psutil

import os

def cpuStats():

        print(sys.version)

        print(psutil.cpu_percent())

        print(psutil.virtual_memory())  # physical memory usage

        pid = os.getpid()

        py = psutil.Process(pid)

        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think

        print('memory GB:', memoryUse)



cpuStats()
# %%timeit

use_cuda = torch.cuda.is_available()

# use_cuda = False



lgr.info("USE CUDA=" + str (use_cuda))

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

Tensor = FloatTensor

# fix seed

seed=17*19

np.random.seed(seed)

torch.manual_seed(seed)

if use_cuda:

    torch.cuda.manual_seed(seed)
data = pd.read_json('../input/train.json')



data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))

data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')





band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)

band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)

full_img = np.stack([band_1, band_2], axis=1)

# Convert the np arrays into the correct dimention and type

# Note that BCEloss requires Float in X as well as in y

def XnumpyToTensor(x_data_np):

    x_data_np = np.array(x_data_np, dtype=np.float32)        

#     print(x_data_np.shape)

#     print(type(x_data_np))



    if use_cuda:

#         lgr.info ("Using the GPU")    

        X_tensor = (torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    

    else:

#         lgr.info ("Using the CPU")

        X_tensor = (torch.from_numpy(x_data_np)) # Note the conversion for pytorch

        

#     print((X_tensor.shape)) # torch.Size([108405, 29])

    return X_tensor





# Convert the np arrays into the correct dimention and type

# Note that BCEloss requires Float in X as well as in y

def YnumpyToTensor(y_data_np):    

    y_data_np=y_data_np.reshape((y_data_np.shape[0],1)) # Must be reshaped for PyTorch!

#     print(y_data_np.shape)

#     print(type(y_data_np))



    if use_cuda:

#         lgr.info ("Using the GPU")            

    #     Y = Variable(torch.from_numpy(y_data_np).type(torch.LongTensor).cuda())

        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor).cuda()  # BCEloss requires Float        

    else:

#         lgr.info ("Using the CPU")        

    #     Y = Variable(torch.squeeze (torch.from_numpy(y_data_np).type(torch.LongTensor)))  #         

        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float        



#     print(type(Y_tensor)) # should be 'torch.cuda.FloatTensor'

#     print(y_data_np.shape)

#     print(type(y_data_np))    

    return Y_tensor
class FullTrainningDataset(torch.utils.data.Dataset):

    def __init__(self, full_ds, offset, length):

        self.full_ds = full_ds

        self.offset = offset

        self.length = length

        assert len(full_ds)>=offset+length, Exception("Parent Dataset not long enough")

        super(FullTrainningDataset, self).__init__()

        

    def __len__(self):        

        return self.length

    

    def __getitem__(self, i):

        return self.full_ds[i+self.offset]

    

validationRatio=0.11    



def trainTestSplit(dataset, val_share=validationRatio):

    val_offset = int(len(dataset)*(1-val_share))

    print ("Offest:" + str(val_offset))

    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, val_offset, len(dataset)-val_offset)
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset



batch_size=128



transformations = transforms.Compose([transforms.ToTensor()])



# train_imgs = torch.from_numpy(full_img_tr).float()

train_imgs=XnumpyToTensor (full_img)

train_targets = YnumpyToTensor(data['is_iceberg'].values)

dset_train = TensorDataset(train_imgs, train_targets)





train_ds, val_ds = trainTestSplit(dset_train)



train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, 

                                            num_workers=1)

val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)



print (train_loader)

print (val_loader)
import math 

def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

    



class CifarResNet(nn.Module):

    def __init__(self, block, n_size, num_classes=1):

        super(CifarResNet, self).__init__()

        self.inplane = 16

        self.conv1 = nn.Conv2d(2, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplane)

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, blocks=2 * n_size, stride=2)

        self.layer2 = self._make_layer(block, 32, blocks=2 * n_size, stride=2)

        self.layer3 = self._make_layer(block, 64, blocks=2 * n_size, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(64, num_classes)

        self.sig=nn.Sigmoid()   

        

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride):



        layers = []

        for i in range(1, blocks):

            layers.append(block(self.inplane, planes, stride))

            self.inplane = planes



        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)



        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)



        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        x = self.sig(x)



        return x



model = CifarResNet(CifarSEBasicBlock, 1, 1)        

print (model)
loss_func=torch.nn.BCELoss()

# NN params

LR = 0.005

MOMENTUM= 0.9

optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=5e-5) #  L2 regularization

if use_cuda:

    lgr.info ("Using the GPU")    

    net.cuda()

    loss_func.cuda()



lgr.info (optimizer)

lgr.info (loss_func)
num_epoches = 20

criterion=loss_func

all_losses = []





model.train()

for epoch in range(num_epoches):

    print('Epoch {}'.format(epoch + 1))

    print('*' * 5 + ':')

    running_loss = 0.0

    running_acc = 0.0

    for i, data in enumerate(train_loader, 1):

        

        img, label = data        

        img = Variable(img)

        label = Variable(label)

                        

        out = model(img)

        loss = criterion(out, label)

        running_loss += loss.data[0] * label.size(0)        

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

                                

        if i % 10 == 0:

            all_losses.append(running_loss / (batch_size * i))

            print('[{}/{}] Loss: {:.6f}'.format(

                epoch + 1, num_epoches, running_loss / (batch_size * i),

                running_acc / (batch_size * i)))

            

    print('Finish {} epoch, Loss: {:.6f}'.format(epoch + 1, running_loss / (len(train_ds))))



torch.save(model.state_dict(), './cnn.pth')
from sklearn.cross_validation import train_test_split



def kFoldValidation(folds): 

    print ('K FOLD VALIDATION ...')

    val_losses = []

    model.eval()

    

    for e in range(folds):

        print ('Fold:' + str(e))        

        data = pd.read_json('../input/train.json')        

        data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))

        data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

        data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')

        band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)

        band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)

        full_img = np.stack([band_1, band_2], axis=1)

        

        X_train,X_val,y_train,y_val=train_test_split(full_img,data['is_iceberg'].values,

                                                   test_size=0.22, 

                                                   random_state=(1+e))

        val_imgs=XnumpyToTensor (X_val)

        val_targets = YnumpyToTensor(y_val)

        

        dset_val = TensorDataset(val_imgs, val_targets)        

        val_loader = torch.utils.data.DataLoader(dset_val, batch_size=batch_size, shuffle=True, 

                                                    num_workers=1)        

        print (val_loader)



        eval_loss = 0

        eval_acc = 0

        for data in val_loader:

            img, label = data



            img = Variable(img, volatile=True)

            label = Variable(label, volatile=True)



            out = model(img)

            loss = criterion(out, label)

            eval_loss += loss.data[0] * label.size(0)



        print('VALIDATION Loss: {:.6f}'.format(eval_loss / (len(dset_val))))

        val_losses.append(eval_loss / (len(dset_val)))

        print()

    

def LeavOneOutValidation(val_loader): 

    print ('Leave one out VALIDATION ...')

    val_losses = []

    model.eval()

        

    print (val_loader)



    eval_loss = 0

    eval_acc = 0

    for data in val_loader:

        img, label = data



        img = Variable(img, volatile=True)

        label = Variable(label, volatile=True)



        out = model(img)

        loss = criterion(out, label)

        eval_loss += loss.data[0] * label.size(0)



    print('Leave on out VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_ds))))

    val_losses.append(eval_loss / (len(val_ds)))

    print()

    print()        

    

LeavOneOutValidation(val_loader)    

# kFoldValidation(10)
df_test_set = pd.read_json('../input/test.json')



df_test_set['band_1'] = df_test_set['band_1'].apply(lambda x: np.array(x).reshape(75, 75))

df_test_set['band_2'] = df_test_set['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

df_test_set['inc_angle'] = pd.to_numeric(df_test_set['inc_angle'], errors='coerce')



df_test_set.head(3)
print (df_test_set.shape)

columns = ['id', 'is_iceberg']

df_pred=pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)

# df_pred.id.astype(int)



for index, row in df_test_set.iterrows():

    rwo_no_id=row.drop('id')    

    band_1_test = (rwo_no_id['band_1']).reshape(-1, 75, 75)

    band_2_test = (rwo_no_id['band_2']).reshape(-1, 75, 75)

    full_img_test = np.stack([band_1_test, band_2_test], axis=1)



    x_data_np = np.array(full_img_test, dtype=np.float32)        

    if use_cuda:

        X_tensor_test = Variable(torch.from_numpy(x_data_np).cuda()) # Note the conversion for pytorch    

    else:

        X_tensor_test = Variable(torch.from_numpy(x_data_np)) # Note the conversion for pytorch

                    

#     X_tensor_test=X_tensor_test.view(1, trainX.shape[1]) # does not work with 1d tensors            

    predicted_val = (model(X_tensor_test).data).float() # probabilities     

    p_test =   predicted_val.cpu().numpy().item() # otherwise we get an array, we need a single float

    

    df_pred = df_pred.append({'id':row['id'], 'is_iceberg':p_test},ignore_index=True)

#     df_pred = df_pred.append({'id':row['id'].astype(int), 'probability':p_test},ignore_index=True)



df_pred.head(5)
# df_pred.id=df_pred.id.astype(int)



def savePred(df_pred):

#     csv_path = 'pred/p_{}_{}_{}.csv'.format(loss, name, (str(time.time())))

#     csv_path = 'pred_{}_{}.csv'.format(loss, (str(time.time())))

    csv_path='sample_submission.csv'

    df_pred.to_csv(csv_path, columns=('id', 'is_iceberg'), index=None)

    print (csv_path)

    

savePred (df_pred)