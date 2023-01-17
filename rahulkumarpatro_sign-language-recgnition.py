# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import time
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
train = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')
test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
train.head()
test.head()
train.shape
test.shape
train.describe()
test.describe()
label_counts  = train["label"].value_counts().sort_index()
label_counts.plot.bar()
train['label'].nunique()
train_fe=train.iloc[:,1:]  # neglecting the label column
train_lab=train['label']  # taking the labels column

test_fe=test.iloc[:,1:]  # neglecting the label column
test_lab=test['label']  # taking the labels column
# converting to numpy 1d array

train_fe_numpy = train_fe.to_numpy()
train_lab_numpy = train_lab.to_numpy()
test_fe_numpy = test_fe.to_numpy()
test_lab_numpy=test_lab.to_numpy()
train_fe.head()
train_fe.iloc[0]
def plot_img(data, label):
    fig, axs = plt.subplots(2,2)
    k = 0
    for i in range(2):
        for j in range(2):        
            axs[i, j].imshow(data[k].reshape(28, 28))            
            axs[i, j].set_ylabel("label:" + str(label[k].item()))   
            k +=4
plot_img(train_fe_numpy, train_lab_numpy)
signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', 
         '6': 'G', '7': 'H', '8': 'I', '10': 'K', '11': 'L', '12': 'M', 
         '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S', 
         '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y','25':'Z'}
reshaped_train = []
for i in train_fe_numpy:
#     print(i)
    reshaped_train.append(i.reshape(1, 28, 28))
train_data = np.array(reshaped_train)

reshaped_test = []
for i in test_fe_numpy:
    reshaped_test.append(i.reshape(1,28,28))
test_data = np.array(reshaped_test)
# train_data
train1,test1,train_label,test_label=train_test_split(train_fe_numpy,train_lab_numpy, test_size=0.2,random_state=42)
print(train1.shape)
print(train_label.shape)
print(test1.shape)
print(test_label.shape)
train_tensor = torch.as_tensor(train1).type(torch.FloatTensor)
train_label = torch.as_tensor(train_label)

test_tensor = torch.as_tensor(test1).type(torch.FloatTensor)
test_label = torch.as_tensor(test_label)
class Convnet(nn.Module):
    
    def __init__(self):
        super(Convnet, self).__init__()
        
        
        self.conv1=nn.Conv2d(1,50,kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
         # L1 ImgIn shape=(?, 28, 28, 1)      # (n-f+2*p/s)+1
        #    Conv     -> (?, 24, 24, 50)
        #    Pool     -> (?, 12, 12, 50)
        
        
        self.conv2 = nn.Conv2d(50,60, kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        # L2 ImgIn shape=(?, 12, 12, 50)
        #    Conv      ->(?, 8, 8, 60)
        #    Pool      ->(?, 4, 4, 60)
        
        
        self.conv3 = nn.Conv2d(60, 80,  kernel_size = 3)
        # L3 ImgIn shape=(?, 4, 4, 60)
        #    Conv      ->(?, 2, 2, 80)
       
        
        
        self.batch_norm1 = nn.BatchNorm2d(50)
        self.batch_norm2 = nn.BatchNorm2d(60)
        
#         self.dropout1 = nn.Dropout2d()
        
        # L4 FC 2*2*80 inputs -> 250 outputs
        self.fc1 = nn.Linear(80*2*2, 250) 
        self.fc2 = nn.Linear(250, 25)
        
        
    def forward(self,x):
        x=self.conv1(x)
        x = self.batch_norm1(x)
        x=F.relu(x)
        x=self.pool1(x)
        
        x=self.conv2(x)
        x = self.batch_norm2(x)
        x=F.relu(x)
        x=self.pool2(x)
        
        x=self.conv3(x)
        x=F.relu(x)
        
        x = x.view(-1,80*2*2)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = F.log_softmax(x, dim=1)
        
        return x     
net=Convnet()

net.eval()  
def get_accuracy(predictions, true_labels):
    _, predicted = torch.max(predictions, 1)
    corrects = (predicted == true_labels).sum()
    accuracy = 100.0 * corrects/len(true_labels)
    return accuracy.item()
def training(loader,model,epochs,criteria,optimizer):
    
    tr_accuracy,tr_loss= [], []
    
    model.train()
   
    
    for epoch in range(epochs):
        
        train_loss = 0 
        train_accuracy = 0
        total_batch = 0
        
        t0=time.time()
        for data,labels in loader:
             # zero the parameters gradient to not accumulate gradients from previous iteration
            optimizer.zero_grad()
            
            
#             print(data.shape)
#             print(labels.shape)
#             put data into the model
#             model(data.permute(50,5,5,1))
#             data=data.reshape(50,60,5,5)
            predictions = net(data)
            
            # calculating loss
            loss = criterion(predictions, labels)
            
            # calculating accuracy
            accuracy = get_accuracy(predictions, labels)
            
            # computing gradients
            loss.backward()
            
            # changing the weights
            optimizer.step()
            
            total_batch+=1
            train_loss += loss.item()
            train_accuracy += accuracy
            
        tfin= time.time()-t0   
        acc=train_accuracy/total_batch  
        loss=train_loss/total_batch
        tr_accuracy.append(acc)
        tr_loss.append(loss)
        
        print("Epoch {}/{}".format(epoch+1,epochs),"Training Loss: {}".format(loss),"Training Accuracy: {}".format(acc),"Time: {} seconds".format(tfin))
        
    return tr_accuracy, tr_loss   
train_tensor.shape
train_tensor=train_tensor.reshape(21964,1,28,28) / 255
train_tensor.shape
!pip install torchsummary
from torchsummary import summary
train_dataset = torch.utils.data.TensorDataset(train_tensor, train_label)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size =50, shuffle = True)

epochs = 30                                              # setting number of epochs

net = Convnet()                                          # initializing the  network
criterion = nn.CrossEntropyLoss()                        # setting criterion
optimizer = torch.optim.SGD(net.parameters(), lr = 3e-4) # setting optimizer

tr_acc, tr_loss = training(trainloader, net,epochs, criterion, optimizer)
torch.save(net, 'model_trained.pt')
test_tensor.shape
test_tensor=test_tensor.reshape(5491,1,28,28) / 255
val_pred = net(test_tensor)
val_loss = criterion(val_pred, test_label)
val_accuracy = get_accuracy(val_pred, test_label)
 
print("Loss: ", (val_loss.item()), "Accuracy: ", (val_accuracy))

# to get class with the maximum score as prediction
_, val_predicted = torch.max(val_pred.data,1)            

skplt.metrics.plot_confusion_matrix(test_label, val_predicted, figsize=(20,20))