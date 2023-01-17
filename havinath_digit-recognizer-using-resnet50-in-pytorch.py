import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.functional as F
from sklearn.model_selection import train_test_split
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
sample_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
x_train, x_val, y_train, y_val = train_test_split(train_data.values[:, 1:], train_data.values[:, 0], test_size=0.2) 
batch_size = 16

x_train_tensor = torch.from_numpy(x_train.astype(np.float32)/255).view(-1, 1, 28, 28)
y_train_tensor = torch.from_numpy(y_train)
x_val_tensor = torch.from_numpy(x_val.astype(np.float32)/255).view(-1, 1, 28, 28)
y_val_tensor = torch.from_numpy(y_val)
test_tensor = torch.from_numpy(test_data.values[:,:].astype(np.float32)/255).view(-1, 1, 28, 28)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(test_tensor)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
resnet_cls = models.resnet50(pretrained=True, progress=True) #Load Pretrained ResNet50

#Convert activations of ResNet50 to LeakyReLU
resnet_cls.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet_cls.relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer1[0].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer1[1].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer1[2].relu = nn.LeakyReLU(inplace=True)

resnet_cls.layer2[0].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer2[1].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer2[2].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer2[3].relu = nn.LeakyReLU(inplace=True)

resnet_cls.layer3[0].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer3[1].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer3[2].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer3[3].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer3[4].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer3[5].relu = nn.LeakyReLU(inplace=True)


resnet_cls.layer4[0].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer4[1].relu = nn.LeakyReLU(inplace=True)
resnet_cls.layer4[2].relu = nn.LeakyReLU(inplace=True)

    
class ResNet50(nn.Module):
    def __init__(self,num_outputs):
        super(ResNet50,self).__init__()
        self.resnet = resnet_cls
         #Remove and manually add FC layers so that image features can be extracted after training
        self.resnet.fc = nn.Sequential() 
        self.linear = nn.Linear(2048, num_outputs,bias=True)
        
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True
            
    def forward(self,x):
        resout = self.resnet(x)
        x = self.linear(resout)
        return x
    
NeuralNet = ResNet50(num_outputs = 10) #Initialise model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")


NeuralNet = NeuralNet.to(device)
optimizer = optim.Adam(NeuralNet.parameters(),lr = 0.0001, weight_decay=1e-5)
loss_func = torch.nn.BCELoss()
best_f1 = np.NINF
start_epoch = 0
train_loss = []
val_loss = []
val_fone=[]
criterion = nn.CrossEntropyLoss()
CNN_EPOCHS =15
import random
import time
import datetime
from sklearn.metrics import accuracy_score
import csv
import copy
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


starttime = datetime.datetime.now()
print("Training started @ "+starttime.strftime("%d-%m-%Y %H:%M:%S"))

#Uncomment to load a check point
#NeuralNet, optimizer_tmp, start_epoch = load_ckp("/content/drive/My Drive/Code/algorithm/Checkpoint-16-05-2020-16-55.pt", NeuralNet, optimizer)


for epoch in range(start_epoch,CNN_EPOCHS):
        start_time = time.time()
        NeuralNet.train()
        running_loss = 0.0
        counter = 0
        prev_lr=0
        prev_loss = 0
        for idx1,(images_batch, labels_batch) in enumerate(train_loader):
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)            
            optimizer.zero_grad()            
            with torch.set_grad_enabled(True):
                pred_batch = NeuralNet(images_batch)
                loss = criterion(pred_batch, labels_batch)
                
            #_, predicted = pred_batch.max(1)
            #correct += (predicted == labels_batch).sum().item()
            #total += labels_batch.size(0)
            #acc = correct / total
        
            current_lr = optimizer.param_groups[0]['lr']
            loss.backward()
            optimizer.step()
            prev_lr = current_lr
            prev_loss = loss
            running_loss += loss.item() * images_batch.size(0)
            counter+=1
        trian_epoch_loss = running_loss / len(train_loader.dataset)
              
        NeuralNet.eval()
        running_loss = 0.0
        counter = 0
        all_output_labels = []
        all_preds = []
        for images_batch, labels_batch in val_loader:
            all_output_labels.extend(labels_batch.numpy().tolist())
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()            
            with torch.set_grad_enabled(False):
                pred_batch = NeuralNet(images_batch)
                _, predicted = pred_batch.max(1)
                all_preds.extend(predicted.detach().cpu().numpy().tolist())
                #correct += (predicted == labels_batch).sum().item()
                #total += labels_batch.size(0)
                #acc = correct / total
                loss = criterion(pred_batch, labels_batch)
                #loss = loss_func(pred_batch,labels_batch)    
            running_loss += loss.item() * images_batch.size(0)
            counter+=1
        val_epoch_loss = running_loss / len(val_loader.dataset)
        val_f1_score = accuracy_score(all_output_labels, all_preds)#cal_f1score(all_preds, all_output_labels)
        if val_f1_score > best_f1:            
            print("Info : model accuracy Improved from {:.8f} to {:.8f}".format(best_f1,val_f1_score))
            best_f1 = val_f1_score
            best_model_wts = copy.deepcopy(NeuralNet.state_dict())       
        elapsed_time = time.time()-start_time
        train_loss.append(trian_epoch_loss)
        val_loss.append(val_epoch_loss)
        val_fone.append(val_f1_score)
        print("Epoch: {}/{} | Training loss:{:.8f} | Validation loss:{:.8f} | Validation Accuracy:{:.8f} | Time: {:.4f}s".format(epoch+1,
                                                                              CNN_EPOCHS,trian_epoch_loss,val_epoch_loss,val_f1_score,elapsed_time))
endtime = datetime.datetime.now()
print("Training ended @ "+endtime.strftime("%d-%m-%Y %H:%M:%S"))
print("Time taken for training :",endtime-starttime)

plt.figure(figsize=(15,4))
plt.plot(train_loss,label="Training loss",color="green")
plt.plot(val_loss,label="Validation loss",color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs Loss")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(15,4))
plt.plot(val_fone,label="Validation f1-score",color="green")
plt.xlabel("Epoch")
plt.ylabel("f1-score")
plt.title("Epoch vs f1-score")
plt.legend()
plt.grid()
plt.show()

NeuralNet.load_state_dict(best_model_wts)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(15,4))
plt.plot(train_loss,label="Training loss",color="green")
plt.plot(val_loss,label="Validation loss",color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs Loss")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(15,4))
plt.plot(val_fone,label="Validation f1-score",color="green")
plt.xlabel("Epoch")
plt.ylabel("f1-score")
plt.title("Epoch vs f1-score")
plt.legend()
plt.grid()
plt.show()

NeuralNet.load_state_dict(best_model_wts)
NeuralNet.eval()
preds = []
for i, data in enumerate(test_loader):
    image = data[0]
    image = image.cuda()
    output = NeuralNet(image)
    _, prediction = output.max(1)
    preds.extend(prediction.tolist())
sample_submission['Label'] = preds
sample_submission.to_csv('submission.csv', index = False)
torch.save(NeuralNet, '/kaggle/working/resnet50_digitrecognizer.pt')