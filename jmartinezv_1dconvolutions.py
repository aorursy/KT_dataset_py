#Juan David Martínez
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Basic Libraries

from __future__ import division

import pandas as pd
import numpy as np
import pickle

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split


from numpy.lib.stride_tricks import as_strided as ast
import os
import librosa
import librosa.display
import glob 
import skimage

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/urbansound8k/UrbanSound8K.csv")

try:
    file_path = "/kaggle/working/"
    # open a file, where you stored the pickled data
    file = open(file_path + "feature.pickle", 'rb')
    # dump information to that file
    feature = pickle.load(file)

    file = open(file_path + "label.pickle", 'rb')
    # dump information to that file
    label = pickle.load(file)
##Run this the first time
except:
    feature = []
    label = []
    # Function to load files and extract features
    #for i in range(8732):
    bads = 0
    for i in range(8732):
        file_name = '../input/urbansound8k/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, sr=16000)
        #print(sample_rate)
        # We extract mfcc feature from data
        #mels = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        feature.append(X)
        label.append(df["classID"][i])


    with open('feature.pickle', 'wb') as f:
        pickle.dump(feature, f)

    with open('label.pickle', 'wb') as f:
        pickle.dump(label, f)
def chunk_data(data,window_size,overlap_size=0,flatten_inside_window=True):
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1,1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows*window_size - (num_windows-1)*overlap_size,data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(
            data,
            shape=(num_windows,window_size*data.shape[1]),
            strides=((window_size-overlap_size)*data.shape[1]*sz,sz)
            )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows,-1,data.shape[1]))
feats = []
labs = []
bads=[0,0,0]
lens = np.zeros(8732)
for i in range(len(feature)):
    data = feature[i]
    lens[i] = data.shape[0]
    if len(data)<16000:
        aux = np.zeros(16000)
        aux[0:len(data)] = data
        feats.append(aux)
        labs.append(label[i])
        bads[0] = bads[0]+1
    elif (len(data)>=16000 and len(data)<24000):
        aux = data[0:16000]
        feats.append(aux)
        labs.append(label[i])
        bads[1] = bads[1]+1
    elif len(data)>=24000:
        bads[2] = bads[2]+1
        xs = chunk_data(data,16000,overlap_size=8000,flatten_inside_window=True)
        powe = np.sum(xs**2,axis = 1)
        ind = np.argmax(powe)
        feats.append(xs[ind,:])
        labs.append(label[i])    
"""
with open('Data16k.pickle', 'wb') as f:
    pickle.dump(feats, f)

with open('Labels16k.pickle', 'wb') as f:
    pickle.dump(feats, f)
"""


print(len(feats))

X = np.zeros((len(feats),16000))
for i in range(len(feats)):
    X[i,:] = feats[i]
Y = np.array(labs)

print(X.shape,Y.shape)

del feats
del labs
del feature
del label

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
class CNN_1D(torch.nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.conv_layer1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv_layer2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.fc_layer1 = torch.nn.Linear(in_features=64*549, out_features=100, bias=True)
        self.fc_layer2 = torch.nn.Linear(in_features=100, out_features=10, bias=True)
        
    def forward(self, x):
        x = self.conv_layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv_layer2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,0.5)
        x = torch.nn.functional.max_pool1d(x,2)
        x = x.view(x.size(0),-1)
        x = self.fc_layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc_layer2(x)
        log_probs = torch.nn.functional.log_softmax(x, dim=1)
        return log_probs
    
class CNN1D_2(torch.nn.Module):
    def __init__(self):
        super(CNN1D_2, self).__init__()
        self.conv_layer1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv_layer2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.fc_layer1 = torch.nn.Linear(in_features=64*62, out_features=100, bias=True)
        self.fc_layer2 = torch.nn.Linear(in_features=100, out_features=10, bias=True)
        
    def forward(self, x):
        x = self.conv_layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv_layer2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,0.5)
        x = torch.nn.functional.max_pool1d(x,2)
        x = x.view(x.size(0),-1)
        x = self.fc_layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc_layer2(x)
        log_probs = torch.nn.functional.log_softmax(x, dim=1)
        return log_probs
class CNN1D_3(torch.nn.Module):
    def __init__(self):
        super(CNN1D_3, self).__init__()
        self.conv_layer1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv_layer2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.fc_layer1 = torch.nn.Linear(in_features=64*7998, out_features=100, bias=True)
        self.fc_layer2 = torch.nn.Linear(in_features=100, out_features=10, bias=True)
        
    def forward(self, x):
        x = self.conv_layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv_layer2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,0.5)
        x = torch.nn.functional.max_pool1d(x,2)
        x = x.view(x.size(0),-1)
        x = self.fc_layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc_layer2(x)
        log_probs = torch.nn.functional.log_softmax(x, dim=1)
        return log_probs

class CNN1D_4(torch.nn.Module):
    def __init__(self):
        super(CNN1D_4, self).__init__()
        self.conv_layer1 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride = 2)
        self.conv_layer2 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride = 2)
        self.conv_layer3 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride = 2)
        self.conv_layer4 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride = 2)
        self.fc_layer1 = torch.nn.Linear(in_features=8*128, out_features=128, bias=True)
        self.fc_layer2 = torch.nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc_layer3 = torch.nn.Linear(in_features=64, out_features=10, bias=True)
        
    def forward(self, x):
        x = self.conv_layer1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool1d(x,kernel_size = 8,stride = 8)
        
        x = self.conv_layer2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool1d(x,kernel_size = 8,stride = 8)
        
        x = self.conv_layer3(x)
        x = torch.nn.functional.relu(x)
        
        x = self.conv_layer4(x)
        x = torch.nn.functional.relu(x)
        
        x = x.view(x.size(0),-1)
        
        x = self.fc_layer1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,p=0.5)
        
        x = self.fc_layer2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x,p=0.5)
        
        x = self.fc_layer3(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        
        return x
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,train_size=0.9, test_size=0.1, random_state = 42)
Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain,Ytrain,train_size=0.9, test_size=0.1, random_state = 42)

train_data = torch.utils.data.TensorDataset(torch.from_numpy(Xtrain), torch.from_numpy(Ytrain))
val_data = torch.utils.data.TensorDataset(torch.from_numpy(Xval), torch.from_numpy(Yval))


batch_size = 100
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)
val_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=False)

model = CNN1D_4()
model.to(device)

#criterion = torch.nn.CrossEntropyLoss() # reduction='sum' created huge loss value
criterion = torch.nn.NLLLoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_episodes = 100


torch.cuda.empty_cache()
model.train()
for t in range(train_episodes):
    for i, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        #yn = torch.argmax(y, dim=1)
        yn = y
        x = x.unsqueeze(1)
        prob = model(x.to(device).float()) 
        loss = criterion(prob, yn.to(device).long())  
        loss.backward()
        optimizer.step()   
        
        if i %20 == 0:
            print(f'epoch : [{t+1}/{train_episodes}], step: [{i}/{len(train_loader)}], loss_train: {loss.item():.4f} ')

loss_bm = 1000000
model_path = '/kaggle/working/best_model'
torch.cuda.empty_cache()
for t in range(train_episodes):
    model.train()
    loss_train = 0
    loss_val = 0
    for i, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.unsqueeze(1)
        prob = model(x.to(device).float())
        loss = criterion(prob, y.to(device).long())  
        loss_train += loss.item()
        loss.backward()
        optimizer.step() 
    
    model.eval()
    for i, (x,y) in enumerate(val_loader):
        x = x.unsqueeze(1)
        prob = model(x.to(device).float())
        loss_v = criterion(prob, y.to(device).long())
        loss_val += loss_v.item()
        
    if(loss_v<loss_bm):
        torch.save(model.state_dict(), model_path)

        
    print(f'epoch : [{t+1}/{train_episodes}], loss_train: {loss_train/len(train_loader):.4f}, loss_val: {loss_val/len(val_loader):.4f} ')

model = CNN1D_4()
model.load_state_dict(torch.load(model_path))
model.to(device)

#Evaluate the model
test_data = torch.utils.data.TensorDataset(torch.from_numpy(Xtest), torch.from_numpy(Ytest))
batch_size = 1
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=False)
model.eval()
torch.cuda.empty_cache()
predi = []

acc = []
yn = []
for i, (x,y) in enumerate(test_loader):
    #yn = torch.argmax(y, dim=1)
    yn.append(y.cpu())
    x = x.unsqueeze(1)
    with torch.no_grad():
        prob = model(x.to(device).float()) 
    pred = torch.argmax(prob, dim=1)
    pred = pred.cpu()
    predi.append(pred)

predi = np.array(predi)  
yn = np.array(yn)
acc = np.mean(np.equal(predi, yn))
print(acc)

