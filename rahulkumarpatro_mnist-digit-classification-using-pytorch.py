# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import time
import torch.nn.functional as F 
train = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
test = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')
train.head()
test.head()
train.shape
test.shape
train.describe()
test.describe()
label_counts  = train["label"].value_counts().sort_index()
label_counts.plot.bar()
train_fe=train.iloc[:,1:]  # neglecting the label column
train_lab=train['label']  # taking the labels column

# converting to numpy 1d array

train_fe_numpy = train_fe.to_numpy()
train_lab_numpy = train_lab.to_numpy()
test_numpy = test.to_numpy()
#train_fe_numpy
train_fe_numpy.shape
test_numpy
def plot_img(data, label):
    fig, axs = plt.subplots(2,2)
    k = 0
    for i in range(2):
        for j in range(2):        
            axs[i, j].imshow(data[k].reshape(28, 28))            
            axs[i, j].set_ylabel("label:" + str(label[k].item()))   
            k +=4
plot_img(train_fe_numpy, train_lab_numpy)
train,test,train_label,test_label=train_test_split(train_fe_numpy,train_lab_numpy, test_size=0.2,random_state=42)
print(train.shape)
print(train_label.shape)
print(test.shape)
print(test_label.shape)
unique, counts_train = np.unique(train_label, return_counts=True)
plt.subplot(1, 2, 1)
plt.bar(unique, counts_train/len(train_label))
unique, counts_val = np.unique(test_label, return_counts=True)
plt.subplot(1, 2, 2)
plt.bar(unique, counts_val/len(test_label))
# train_all_tensor = torch.as_tensor(train_fe_numpy).type(torch.FloatTensor)
# train_all_label_tensor = torch.as_tensor(train_lab_numpy)
# test_tensor = torch.as_tensor(test_numpy).type(torch.FloatTensor)

train_tensor = torch.as_tensor(train).type(torch.FloatTensor)
train_label = torch.as_tensor(train_label)

test_tensor = torch.as_tensor(test).type(torch.FloatTensor)
test_label = torch.as_tensor(test_label)
plot_img(train_tensor,train_label)
# train_tensor
# test_tensor
class FNet(nn.Module):
    def __init__(self):
         super(FNet, self).__init__()  
         
         self.l1=nn.Linear(in_features=784, out_features=600)   # 784 inputs, connects to hidden layer with 600 nodes
         
#          self.relu = nn.ReLU()
            
         self.l2=nn.Linear(in_features=600, out_features=500)   #  600 nodes connects to hidden layer with 500 nodes
         
#          self.relu = nn.ReLU()   
        
         self.l3 = nn.Linear(in_features=500, out_features=250)  # 500 nodes connects to hidden layer with 250 nodes
            
#          self.relu = nn.ReLU()
        
         self.l4 = nn.Linear(in_features=250, out_features=10)   # 250 nodes connects to hidden layer and output layer of 10 nodes
        
    def forward(self, x):
        x = x.view(-1,784)   # Putting all the entries of the image in the vector     
        x = F.relu(self.l1(x))     # Input x into first layer and apply a ReLU
                                    # to the nodes in this layer
        x = F.relu(self.l2(x))        
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x               
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
            
            # put data into the model
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
        
batch_size = 128                                     
train_dataset = torch.utils.data.TensorDataset(train_tensor, train_label)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

epochs = 60                                              # setting number of epochs

net = FNet()                                              # initializing the  network
criterion = nn.CrossEntropyLoss()                         # setting criterion
optimizer = torch.optim.SGD(net.parameters(), lr = 3e-4) # setting optimizer

tr_acc, tr_loss = training(trainloader, net,epochs, criterion, optimizer)
def curves(epochs,loss,acc):
    
    iters=range(1,epochs+1)
    fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(15,5))
    fig.suptitle('Training Curve')
    ax1.plot(iters, loss)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Training Loss")
    ax2.plot(iters, acc, color = 'g')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Training Accuracy")
    plt.show()
curves(epochs,tr_loss,tr_acc)
net.eval()  
def get_accuracy(predictions, true_labels):
    _, predicted = torch.max(predictions, 1)
    corrects = (predicted == true_labels).sum()
    accuracy = 100.0 * corrects/len(true_labels)
    return accuracy.item()
val_pred = net(test_tensor)
val_loss = criterion(val_pred, test_label)
val_accuracy = get_accuracy(val_pred, test_label)
 
print("Loss: ", (val_loss.item()), "Accuracy: ", (val_accuracy))

# to get class with the maximum score as prediction
_, val_predicted = torch.max(val_pred.data,1)            

skplt.metrics.plot_confusion_matrix(test_label, val_predicted, figsize=(10,10))
val
plot_img(test,val_predicted)