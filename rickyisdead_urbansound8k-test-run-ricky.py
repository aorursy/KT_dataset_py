# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import librosa
import matplotlib.pyplot as plt
# Mel Spectrogram preprocessing / use with model 1

def extract_features(file_name):
    try:
        audio,sr = librosa.load(file_name, res_type='kaiser_fast')
        
        # pad audio files less than 3s, cut more than 3s
        if audio.shape[0] < 4*sr:
            audio = np.pad(audio, int(np.ceil((4*sr-audio.shape[0])/2)), mode='reflect')
        else:
            audio = audio[:4*sr]
        
        mel_spec = librosa.feature.melspectrogram(audio, n_mels=128, fmin=20, fmax=8300)
        db_mel_spec = librosa.power_to_db(mel_spec, top_db=80)
    
    except Exception as e:
        print("Error while parsing file: ", file)
        return None
    
    return db_mel_spec
# # MFCCs mean preprocessing / use with model 2

# def extract_features(file_name):
#     try:
#         audio,sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=64)
        
#         mfccsscaled = np.mean(mfccs.T, axis=0)
        
#         mfccsscaled = np.reshape(mfccsscaled, (8,8))
    
#     except Exception as e:
#         print("Error while parsing file: ", file)
#         return None
    
#     return mfccsscaled
# # MFCCs padded preprocessing / use with Model 3

# def extract_features(file_name):
#     try:
#         audio,sample_rate = librosa.load(file_name, res_type='kaiser_fast')
#         max_pad = 174
        
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#         pad_width = max_pad - mfccs.shape[1]
        
#         mfccspadded = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
    
#     except Exception as e:
#         print("Error while parsing file: ", file)
#         return None
    
#     return mfccspadded
#Model 1

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 24, 3, padding=0)
        self.conv1_bn = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, 3, padding=0)
        self.conv2_bn = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 64, 3, padding=0)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=0)
        self.conv4_bn = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(5376, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), (2,4))
        x = F.dropout(x, p = 0.2)
        
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2)
        x = F.dropout(x, p = 0.2)
        
        x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), 2)
        x = F.dropout(x, p = 0.2)
        
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.dropout(x, p = 0.2)
        
        x = torch.flatten(x, start_dim = 1)
        
        x = F.softmax(self.fc1(x), dim = 0)
        return x
# # Model 2

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
        
#         self.conv1 = nn.Conv2d(1, 32, 3)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.conv3 = nn.Conv2d(64, 128, 3)
        
#         self.fc1 = nn.Linear(512, 128)
#         self.fc2 = nn.Linear(128, 10)
    
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.dropout(x, p = 0.2)
        
#         x = F.relu(self.conv2(x))
#         x = F.dropout(x, p = 0.2)
        
#         x = F.relu(self.conv3(x))
#         x = F.dropout(x, p = 0.2)
        
#         x = torch.flatten(x, start_dim = 1)
#         x = torch.tanh(self.fc1(x))
#         x = F.softmax(self.fc2(x), dim = 0)
#         return x
# #Model 3

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
        
#         self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
#         self.conv1_bn = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.conv2_bn = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3_bn = nn.BatchNorm2d(64)
#         self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv4_bn = nn.BatchNorm2d(128)
#         self.conv5 = nn.Conv2d(128, 196, 3, padding = 1)
#         self.conv5_bn = nn.BatchNorm2d(196)
        
#         self.fc1 = nn.Linear(980, 10)
    
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), 2)
#         x = F.dropout(x, p = 0.2)
        
#         x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), 2)
#         x = F.dropout(x, p = 0.2)
        
#         x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), 2)
#         x = F.dropout(x, p = 0.2)
        
#         x = F.max_pool2d(F.relu(self.conv4_bn(self.conv4(x))), 2)
#         x = F.dropout(x, p = 0.2)
        
#         x = F.avg_pool2d(F.relu(self.conv5_bn(self.conv5(x))), 2)
#         x = F.dropout(x, p = 0.2)
        
#         x = torch.flatten(x, start_dim = 1)
        
#         x = torch.tanh(self.fc1(x))
#         x = F.softmax((x), dim = 0)
#         return x
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance (m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
def create_loader(inputdf, batch):
    x = np.array(inputdf.feature.tolist())
    x = torch.from_numpy(x)

    y = torch.tensor(inputdf['classID'].values)

    data = TensorDataset(x, y)
    loader = DataLoader(data, batch_size = batch, shuffle = True)
    
    return loader
def train(model, train_loader, MAX_EPOCHS, l_rate):    
    
    criterion = nn.CrossEntropyLoss()
    model.train()
#     model.apply(weights_init)
    
    for epoch in range(MAX_EPOCHS):
        total = 0.0
        correct = 0.0
        
#         if epoch % 30 == 0:
#             optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=0.0001)
#             l_rate = l_rate / 10

        # note: momentum SGD > Adam for Model 1 + Mel-Spec
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=l_rate, nesterov=True, weight_decay=0.0001)
        
        
        
        for local_batch, local_labels in train_loader:
            optimizer.zero_grad()
            
            local_batch = local_batch.to(device, dtype = torch.float32)
            local_labels = local_labels.to(device, dtype = torch.long)

            local_batch = local_batch.unsqueeze(dim=1)

            local_outputs = model(local_batch)
            
            _, predicted = torch.max(local_outputs.data, 1)
            total += local_labels.size(0)
            correct += (predicted == local_labels).sum().item()
            
            loss = criterion(local_outputs, local_labels)
            loss.backward()
            optimizer.step()
           
        if epoch % 10 == 0:
             print('Epoch ', epoch, ' Acc: %d %%' % (100 * correct / total),)

    return model
def validate(model, test_loader):

    total = 0.0
    correct = 0.0
    
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = images.unsqueeze(dim=1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (100 * correct / total)
    
    return accuracy
# Data import segment

fulldatasetpath = '../input/urbansound8k'

metadata = pd.read_csv(fulldatasetpath + '/UrbanSound8K.csv')

CLASS_NAMES = ['air conditioner', 'car horn', 'children playing', 'dog bark', 'drilling', 'engine idling', 'gun shot', 'jackhammer', 'siren', 'street music']

features = []

i = 0

for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
    class_id = row["classID"]
    fold = row["fold"]
    data = extract_features(file_name)
    
#     print(data.shape)
    
    features.append([data, class_id, fold])
    
#     #limit for compute constraints
#     i += 1
#     if (i > 100):
#         break

# keep fold feature to evaluate according to UrbanSound specification
featuresdf = pd.DataFrame(features, columns=['feature','classID','fold'])
print(featuresdf.head())
print(featuresdf['feature'][0].shape)
# save data to csv, need to implement converting csv entries back to array

featuresdf.to_csv('mel_spec.csv', index=False)

# Random train/test split
from sklearn.model_selection import train_test_split

x, y = np.array(featuresdf.feature.tolist()), np.array(featuresdf.classID.tolist())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train = torch.from_numpy(x_train)
y_train = torch.tensor(y_train)

train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)

x_test = torch.from_numpy(x_test)
y_test = torch.tensor(y_test)

test_data = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = True)
MAX_EPOCHS = 60

if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')

model = ConvNet()
model = model.to(device)

# 0.001 for Adam, 0.01 for SGD
model = train(model, train_loader, MAX_EPOCHS, l_rate=0.01)

ac = validate(model, test_loader)
# Accs.append(ac)

print('Test Accuracy: ', ac, '%%')
    
# print('10 Fold Cross Validation: ', np.mean(Accs), '%%')
# 10-fold cross validation
MAX_EPOCHS = 60
accs = []

if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')

for fold in range(1,11):

    test_fold = fold

    traindf = featuresdf[featuresdf['fold'] != test_fold]
    testdf = featuresdf[featuresdf['fold'] == test_fold]

    train_loader = create_loader(traindf, 64)
    test_loader = create_loader(testdf, 64)

    model = ConvNet()
    model = model.to(device)

    model = train(model, train_loader, MAX_EPOCHS, l_rate=0.001)

    ac = validate(model, test_loader)
    accs.append(ac)

    print('Fold ', fold, ' Accuracy: ', ac, '%%')

print('10 Fold Cross Validation: ', np.mean(accs), '%%')