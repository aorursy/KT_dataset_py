a = []
while(1):
    a.append('1')
from google.colab import files
!pip install -q kaggle
uploaded = files.upload()
%cd ..
!mkdir root/.kaggle/
!cp content/kaggle.json root/.kaggle/kaggle.json
!kaggle competitions download -c 11-785-s20-hw2p2-classification
!unzip 11-785-hw2p2-s20.tgz.zip
!tar zxvf 11-785-hw2p2-s20.tgz
%cd 11-785hw2p2-s20
!unzip \*.zip
%cd 11-785hw2p2-s20
!ls
%cd train_data/large
!ls -1 | wc -l
%cd ..
%cd ..
%cd train_data/medium
!ls -1 | wc -l
%cd ..
%cd ..
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torchvision.models as models
import numpy as np
import pandas as pd
from PIL import Image

import sys
import torch.optim as optim

from torch.utils import data
from torch.utils.data import DataLoader, Dataset, TensorDataset

import matplotlib.pyplot as plt
import time

cuda = torch.cuda.is_available()
cuda
num_workers = 8

data_transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.ImageFolder(root='train_data/medium/', transform=data_transform)
val_data = datasets.ImageFolder(root='validation_classification/medium/', transform=data_transform)
    
# Training
train_loader_args = dict(shuffle=True, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=True, batch_size=64)
train_loader = torch.utils.data.DataLoader(train_data, **train_loader_args) 

# Validating
val_loader_args = dict(shuffle=False, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)
val_loader = torch.utils.data.DataLoader(val_data, **val_loader_args)
print(len(train_data.classes))
from google.colab import drive
drive.mount('/content/gdrive')
class CNN(nn.Sequential):
    
    def __init__(self, c_in, c_out, s, k=3):
        
        super(CNN, self).__init__(
            nn.Conv2d(c_in, c_out, k, s, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=True)
        )
    
class DepthwiseCNN(nn.Sequential):
    
    def __init__(self, c_in, c_out, s, k=3):
        
        super(DepthwiseCNN, self).__init__(
            nn.Conv2d(c_in, c_out, k, s, 1, groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=True)
        )
        
class Expansion(nn.Sequential):
    
    def __init__(self, c_in, c_out, k=1, s=1):
        
        super(Expansion, self).__init__(
            nn.Conv2d(c_in, c_out, k, s, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=True)
        )

class Projection(nn.Sequential):
    
    def __init__(self, c_in, c_out, k=1, s=1):
        
        super(Projection, self).__init__(
            nn.Conv2d(c_in, c_out, k, s, bias=False),
            nn.BatchNorm2d(c_out)
        )

class BottleNeck(nn.Module):
    
    def __init__(self, c_in, c_out, s, t):
        
        super(BottleNeck, self).__init__()
        self.invertedResidual = (c_in == c_out) and (s == 1)
        
        hidden = t*c_in
        self.layers = []
        self.layers.append(Expansion(c_in, hidden))
        self.layers.append(DepthwiseCNN(hidden, hidden, s))
        self.layers.append(Projection(hidden, c_out))
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        if self.invertedResidual:
            return x + self.layers(x)
        else:
            return self.layers(x)
        
class MobileNetV2(nn.Module):
    
    def __init__(self, params):
        
        super(MobileNetV2, self).__init__()

        self.layers = []

        c_in = 3
        for b in range(len(params)):
            t, c_out, n, s = params[b]
            if t is None:
                self.layers.append(CNN(c_in, c_out, s))
                c_in = c_out
            else:
                for i in range(n):
                    if i == 0:
                        self.layers.append(BottleNeck(c_in, c_out, s, t))
                    else:
                        self.layers.append(BottleNeck(c_in, c_out, 1, t))
                    c_in = c_out

        self.layers.append(nn.AvgPool2d(8))
        self.layers.append(nn.Flatten())

        self.layers.append(nn.Linear(c_in, 1000))
        self.layers.append(nn.BatchNorm1d(1000))
        self.layers.append(nn.ReLU(inplace = True))

        self.layers = nn.Sequential(*self.layers)
        
        self.fc1 = nn.Linear(1000, 2300)
        self.fc2 = nn.Linear(1000, 60)

    def forward(self, x):
        x = self.layers(x)
        label = self.fc1(x)
        embedding = self.fc2(x)
        return label, embedding
        
    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight.data)
    
params = [(None,32,1,1), 
          (1,16,1,1), 
          (3,24,2,2), #16 #1
          (3,32,3,1), #2
          (3,64,4,2), #8 #1
          (3,96,3,1),
          (3,160,3,1), #2
          (3,320,1,1),
          (None,1280,1,1)
          ]

#creating model
#model = BasicCNNModule()
model = MobileNetV2(params)
# save only model parameters
#PATH = "./classifier_v2.pt"
#torch.save(model.state_dict(), PATH)

# load a saved model parameters
model_save_name = 'classifier_v2.pt'
path = F"/content/gdrive/My Drive/{model_save_name}"

model.load_state_dict(torch.load(path))

## Less optimised approaches ->
# saving the entire model
#torch.save(model, PATH)
device = torch.device("cuda" if cuda else "cpu")
print(device)
model.to(device)
params = [(None,32,1,1), 
          (1,16,1,1), 
          (3,24,2,2), #16 #1
          (3,32,3,1), #2
          (3,64,4,2), #8 #1
          (3,96,3,1),
          (3,160,3,1), #2
          (3,320,1,1),
          (None,1280,1,1)
          ]

#creating model
model = MobileNetV2(params)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)
#optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 2, factor = 0.8) #0.1
device = torch.device("cuda" if cuda else "cpu")
print(device)
model.to(device)
print(model)
def train_epoch(model, train_loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device) # all data & model on same device

        outputs, embedding = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()

        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        #if batch_idx % 100 == 0:
          #print(batch_idx)
        
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    acc = (correct_predictions/total_predictions)*100.0
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    print('Training Accuracy: ', acc, '%')
    return running_loss, acc
def test_model(model, val_loader, criterion):

    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(val_loader):   
            data = data.to(device)
            target = target.to(device)

            outputs, embedding = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()

          
        running_loss /= len(val_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc

#start training for 10 epochs
n_epochs = 40
Train_loss = []
Val_loss = []
Val_acc = []

for i in range(n_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = test_model(model, val_loader, criterion)
    Train_loss.append(train_loss)
    Val_loss.append(val_loss)
    Val_acc.append(val_acc)
    scheduler.step(val_loss)

    model_save_name = 'classifier_v2_1.pt'
    path = F"/content/gdrive/My Drive/{model_save_name}" 
    torch.save(model.state_dict(), path)

    print('='*20)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0045, momentum=0.9, weight_decay=5e-4, nesterov=True)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)
#optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 1, factor = 0.5)
#start training for 10 epochs
n_epochs = 10
Train_loss = []
Train_acc = []
Val_loss = []
Val_acc = []

for i in range(n_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = test_model(model, val_loader, criterion)
    Train_loss.append(train_loss)
    Train_acc.append(train_acc)
    Val_loss.append(val_loss)
    Val_acc.append(val_acc)
    scheduler.step(val_loss)

    if val_acc > best_acc:
      print('stored')
      model_save_name = 'classifier_v2.pt'
      path = F"/content/gdrive/My Drive/{model_save_name}" 
      torch.save(model.state_dict(), path)
      best_acc = val_acc
    
    print('='*20)
CUDA_LAUNCH_BLOCKING=1
with open('test_order_classification.txt') as file:
     orders = [line.strip() for line in file]
class MyDataset(data.Dataset):

    def __init__(self, data_path, order_path):

        self.data_path = data_path
        with open(order_path) as file:
          self.orders = [line.strip() for line in file]
        self.length = len(orders)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        img = Image.open(self.data_path + "/"+ self.orders[index])
        array = data_transform(img)
        return array
test_data = MyDataset('test_classification/medium','test_order_classification.txt')
test_loader_args = dict(shuffle=False, batch_size=1, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)

test_loader = data.DataLoader(test_data, **test_loader_args)
def predict_model(model, test_loader, orders, labels):

    embedding_map = dict()
    results = []
    embeddings = []
    with torch.no_grad():
        model.eval()

        for batch_idx, data in enumerate(test_loader):   
            data = data.to(device)
            #print(data.shape)
            outputs, embedding = model(data)

            _, predicted = torch.max(outputs.data, 1)
            results.append(predicted)
            embeddings.append(embedding)
      
    ans = pd.DataFrame(columns=["Id", "Category"])
    idx = 0
    for i in range(len(results)):
      for j, label in enumerate(results[i]):
        ans = ans.append({'Id': orders[idx], 'Category':labels[label.item()]}, ignore_index=True)
        embedding_map[orders[idx]] = embeddings[idx]
        idx += 1

    ans_csv = ans.to_csv('result.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
    print(ans)
    return embedding_map
#use trained model to predict on test data
results = predict_model(model, test_loader, orders, train_data.classes)
files.download('result.csv')
class MyDataset(data.Dataset):

    def __init__(self, data_path, order_path):

        self.data_path = data_path
        with open(order_path) as file:
          self.orders = [line.strip() for line in file]
        self.length = len(self.orders)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        imgs = self.orders[index].split(' ')
        img1 = Image.open(self.data_path + "/"+ imgs[0])
        img2 = Image.open(self.data_path + "/"+ imgs[1])

        trans =  transforms.ToTensor()
        array1 = trans(img1)
        array2 = trans(img2)

        return array1, array2
test_data = MyDataset('test_verification/','test_trials_verification_student.txt')
test_loader_args = dict(shuffle=False, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)

test_loader = data.DataLoader(test_data, **test_loader_args)
with open('test_trials_verification_student.txt') as file:
     Vorders = [line.strip() for line in file]
  
print(Vorders[0].split(' '))
%cd test_verification
!ls -1 | wc -l
def train_model(model, array):

  with torch.no_grad():
    model.eval()
    array = array.to(device)
    label, embedding = model(array)
    return label.cpu()
#trials = []
scores = []

for idx, tup in enumerate(test_loader):

    #imgs = Vorders[idx].split(' ')
    arr1, arr2 = tup

    embed1 = train_model(model, arr1)
    embed2 = train_model(model, arr2)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    score = cos(embed1, embed2)
    score = score.flatten()
    scores.extend(score.tolist())

    if idx % 100 == 0:
      print(idx)
        
trials = [Vorders[i] for i in range(len(Vorders))]
print(len(scores))
print(len(trials))
df1 = pd.DataFrame(trials)
df2 = pd.DataFrame(scores)
df = pd.concat([df1, df2], axis = 1)
df.head()
df.columns = ['trial','score']
df.head()
model_save_name = 'verification.csv'
path = F"/content/gdrive/My Drive/{model_save_name}" 
df_csv = df.to_csv(path, index = None, header=True) #Don't forget to add '.csv' at the end of the path
len(df)
class BasicCNNModule(nn.Module):
    def __init__(self):
        super(BasicCNNModule, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.fc1 = nn.Linear(512 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2300)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
print(BasicCNNModule())
class MobileNetV1(nn.Module):

  def __init__(self):
    super(MobileNetV1, self).__init__()

    self.layers = []
    self.layers += self.Conv(3,32,3,1)
    self.layers += self.DepthSepConv(32, 64, 3, 1)
    self.layers += self.DepthSepConv(64, 128, 3, 2)       
    self.layers += self.DepthSepConv(128, 128, 3, 1)       
    self.layers += self.DepthSepConv(128, 256, 3, 1)    
    self.layers += self.DepthSepConv(256, 256, 3, 1) 
    self.layers += self.DepthSepConv(256, 512, 3, 2)      
    self.layers += self.DepthSepConv(512, 512, 3, 1)   
    #self.layers += self.DepthSepConv(512, 512, 3, 1)      
    #self.layers += self.DepthSepConv(512, 512, 3, 1)       
    #self.layers += self.DepthSepConv(512, 512, 3, 1)       
    #self.layers += self.DepthSepConv(512, 512, 3, 1)       
    self.layers += self.DepthSepConv(512, 1024, 3, 1)     
    self.layers += self.DepthSepConv(1024, 1024, 3, 1)    
    self.layers.append(nn.AvgPool2d(8))
    self.layers.append(nn.Flatten())
    self.layers.append(nn.Linear(1024, 1000))
    self.layers.append(nn.BatchNorm1d(1000))
    self.layers.append(nn.ReLU(inplace = True))
    self.layers = nn.Sequential(*self.layers)
    
    self.fc1 = nn.Linear(1000, 2300)
    self.fc2 = nn.Linear(1000, 60)
  
  def forward(self, x):
    x = self.layers(x)
    #x = x.view(-1, 1024)
    #x = self.fc1(x)
    #x = self.bn1(x)
    label = self.fc1(x)
    embedding = self.fc2(x)
    return label, embedding

  def Conv(self, c_in, c_out, k, s):

      return [nn.Conv2d(c_in, c_out, k, s),
          nn.BatchNorm2d(c_out),
          nn.ReLU(inplace = True)]
          

  def DepthSepConv(self, c_in, c_out, k, s):

      return [
          #depth-wise
          nn.Conv2d(c_in, c_in, k, s, 1, groups=c_in, bias=False),
          nn.BatchNorm2d(c_in),
          nn.ReLU(inplace=True),
          #point-wise
          nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
          nn.BatchNorm2d(c_out),
          nn.ReLU(inplace=True)
      ]
  
  def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)