!git clone https://github.com/JulienAu/Anomaly_Detection_Tuto.git
!rm -r Anomaly_Detection_Tuto/*.ipynb Anomaly_Detection_Tuto/.git Anomaly_Detection_Tuto/Data/*.gif
from IPython.display import Image

Image(filename="./Anomaly_Detection_Tuto/Data/Autoencoder_structure.png")
import random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

from torch import nn

from torch.autograd import Variable

import warnings



warnings.filterwarnings('ignore')

%matplotlib inline
import logging

handler=logging.basicConfig(level=logging.INFO)

lgr = logging.getLogger(__name__)



#Utilisation des GPUs

#use_cuda = torch.cuda.is_available()

use_cuda = False



FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

Tensor = FloatTensor



#lgr.info("USE CUDA=" + str (use_cuda))
def Rand(num):

    res = []

    for j in range(num):

        res.append(random.random())

    return res







##### series 0 - half sine, half linear

nb_periods=10

nb_points=2600

x = np.linspace(-nb_periods*np.pi, nb_periods*np.pi, nb_points)

x2 = np.linspace(int(-nb_periods*np.pi/3), int(nb_periods*np.pi/3), int(nb_points/3))

aa=list(np.sin(x))

bb=list(np.zeros(int(nb_points/4)))

cc=list(np.sin(x2))

dd=list(np.arange(int(nb_points/4))/int(nb_points/-4))

ee=list(np.arange(int(nb_points/4))/int(nb_points/4))

y=list(np.arange(len(aa+aa+bb+cc+dd+aa)))

# On joint nos 3 patterns

serie_simple=np.array(aa+aa+bb+cc+ee+cc+dd+aa)



# Visualisation 

fig=plt.figure(figsize=(12,4))

plt.plot(serie_simple)

plt.title("A simple series composed of 4 patterns")

plt.show()
fig=plt.figure(figsize=(12,4))

dataframe = pd.DataFrame(serie_simple.astype('float32'))

df_total=dataframe

plt.plot(df_total)



df= dataframe[:6500]

plt.plot(df,label="Apprentissage")



df_test= dataframe[6500:10000]

plt.plot(df_test,label="Test")

plt.legend()

plt.suptitle("Découpage de la série temporelle")

plt.show()
dataset = df.values

dataset_test = df_test.values
def normalize_test(dataset,dataset_test):

    """

    Permet de normaliser les valeurs de la time-series [0;1] par rapport aux valeurs de la période d'apprentissage

    

    Inputs : 

        data = np array 1D correspondant à une série temporelle

    Output : np.array 1D normalisé

    """

    max_value = np.max(dataset)

    min_value = np.min(dataset)

    scalar = max_value - min_value

    dataset = list(map(lambda x: (x-min_value)   / scalar, dataset))

    dataset_test = list(map(lambda x: (x-min_value)   / scalar, dataset_test))

    return dataset , dataset_test



dataset,dataset_test=normalize_test(dataset,dataset_test)

def rolling_window(data,look_back,pattern_indices=[]):

    """

    Inputs : 

        data = np array 1D correspondant à une série temporelle

        pattern_indices = quand ils sont connus, les indices de changements de patterns

        look_back : taille de la fenêtre glissante

    Output : un np.array 2D avec "look_back" columns 

    """

    dataX=[]

    for i in range(len(data)-look_back-1):

        a = data[i:(i+look_back)]

        dataX.append(list(a))



    dataX=np.array(dataX)

    return dataX
look_back=128

data_X=rolling_window(dataset,look_back)

#print(data_X.shape)

data_X=data_X.reshape(-1, 1,look_back)

#print(data_X.shape)



if use_cuda:

        #lgr.info ("Using the GPU")    

        train_x = torch.from_numpy(data_X).float().cuda() # Note the conversion for pytorch    

else:

        #lgr.info ("Using the CPU")

        train_x = torch.from_numpy(data_X).float() # Note the conversion for pytorch

class autoencoder(nn.Module):

    def __init__(self ,input_size):

        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_size, 128),

            nn.ReLU(True),

            nn.Linear(128, 32),

            nn.ReLU(True),

            nn.Linear(32, 12),

            nn.ReLU(True), 

            nn.Linear(12, 3))

        self.decoder = nn.Sequential(

            nn.Linear(3, 12),

            nn.ReLU(True),

            nn.Linear(12, 32),

            nn.ReLU(True),

            nn.Linear(32, 128),

            nn.ReLU(True),

            nn.Linear(128, input_size), 

            nn.Tanh())



    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x
net = autoencoder(look_back)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
if use_cuda:

    lgr.info ("Using the GPU")    

    net.cuda()

    criterion.cuda()
from IPython.display import Image

Image(filename="./Anomaly_Detection_Tuto/Data/mse.png")
net_loss=[]

for e in range(500):

    var_x = Variable(train_x)

    out = net(var_x)

    loss = criterion(out, var_x)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    net_loss.append(loss.item())

    if (e + 1) % 50 == 0:

        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))     
plt.suptitle("MSE during training ")

plt.plot(net_loss)

plt.show()
net = net.eval()

if use_cuda:

    lgr.info ("Using the GPU")    

    net.cuda()
var_data = Variable(train_x)

pred_test = net(var_data)

pred_test = pred_test.view(-1,128)
def reconstruction_viz(train_x,autoencoder,lookback,index=0):

    var_data = Variable(train_x)

    pred_test = net(var_data)

    pred_test = pred_test.view(-1,lookback).data.cpu().numpy()

    data = train_x.view(-1,lookback).data.cpu().numpy()

    fig=plt.figure(figsize=(12,4))

    plt.plot(data[:,1],color="blue",label="Real series")

    plt.plot(pred_test[:,1],color="green",label="Reconstruction")

    plt.legend()

    plt.suptitle("A piece of a curve and its reconstruction by the auto encoder")

    plt.show()



reconstruction_viz(train_x,net,look_back,index=0)
def reconstruction_viz2(train_x,autoencoder,lookback,index=0):

    var_data = Variable(train_x)

    pred_test = net(var_data)

    pred_test = pred_test.view(-1,lookback).data.cpu().numpy()

    data = train_x.view(-1,lookback).data.cpu().numpy()

    ecart= np.abs(np.subtract(data[:,1],pred_test[:,1]))

    

    fig, ax = plt.subplots(nrows=4, sharey=True,figsize=(12,12))

    ax[0].plot(pred_test[:,1], 'r', label='reconstruction')

    ax[0].legend(loc='best')

    ax[1].plot(data[:,1], 'b', label='real')

    ax[1].legend(loc='best')

    ax[2].plot(pred_test[:,1], 'r', label='reconstruction')

    ax[2].plot(data[:,1], 'b', label='real')

    ax[2].legend(loc='best')

    ax[3].plot(ecart, 'g', label='ecart')

    ax[3].legend(loc='best')

    plt.legend()

    plt.suptitle("Result of the autoencoder")

    plt.show()



reconstruction_viz2(train_x,net,look_back,index=0)
data_X_test=rolling_window(dataset_test,look_back)



data_X_test=data_X_test.reshape(-1, 1,look_back)





if use_cuda:

        #lgr.info ("Using the GPU")    

        test_x = torch.from_numpy(data_X_test).float().cuda() # Note the conversion for pytorch    

else:

        #lgr.info ("Using the CPU")

        test_x = torch.from_numpy(data_X_test).float() # Note the conversion for pytorch







var_data_test = Variable(test_x)

pred_test = net(var_data_test)

pred_test = pred_test.view(-1,128)



reconstruction_viz2(test_x,net,look_back,index=0)
df_total= pd.read_json('./Anomaly_Detection_Tuto/Data/serie2.json', lines=True)[['val']]

plt.plot(df_total)



df= pd.read_json('./Anomaly_Detection_Tuto/Data/serie2.json', lines=True)[['val']][3000:6000]

plt.plot(df,label="Train")



df_test= pd.read_json('./Anomaly_Detection_Tuto/Data/serie2.json', lines=True)[['val']][7000:12000]

plt.plot(df_test,label="Test")



plt.legend()

plt.suptitle("The time series")

plt.show()
dataset = df.values

dataset = dataset.astype('float32')



dataset_test = df_test.values

dataset_test = dataset_test.astype('float32')

dataset,dataset_test=normalize_test(dataset,dataset_test)

data_X=rolling_window(dataset,look_back)



data_X=data_X.reshape(-1, 1,look_back)

if use_cuda:

        #lgr.info ("Using the GPU")    

        train_x = torch.from_numpy(data_X).float().cuda() # Note the conversion for pytorch    

else:

        #lgr.info ("Using the CPU")

        train_x = torch.from_numpy(data_X).float() # Note the conversion for pytorch

net = autoencoder(look_back)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.005)



if use_cuda:

    #lgr.info ("Using the GPU")    

    net.cuda()

    criterion.cuda()
net_loss=[]

for e in range(1200):

    var_x = Variable(train_x)

    out = net(var_x)

    loss = criterion(out, var_x)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    net_loss.append(loss.item())

    if (e + 1) % 100 == 0:

        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
plt.suptitle("MSE during training")

plt.plot(net_loss)

plt.show()
net = net.eval()

if use_cuda:

    #lgr.info ("Using the GPU")    

    net.cuda()
var_data = Variable(train_x)

pred_test = net(var_data)

#print(pred_test.shape)



pred_test = pred_test.view(-1,128)

#print(pred_test.shape)
reconstruction_viz(train_x,net,look_back,index=0)
reconstruction_viz2(train_x,net,look_back,index=0)
data_X_test=rolling_window(dataset_test,look_back)



data_X_test=data_X_test.reshape(-1, 1,look_back)





if use_cuda:

        #lgr.info ("Using the GPU")    

        test_x = torch.from_numpy(data_X_test).float().cuda() # Note the conversion for pytorch    

else:

        #lgr.info ("Using the CPU")

        test_x = torch.from_numpy(data_X_test).float() # Note the conversion for pytorch





var_data_test = Variable(test_x)

pred_test = net(var_data_test)

pred_test = pred_test.view(-1,128)



reconstruction_viz2(test_x,net,look_back,index=0)