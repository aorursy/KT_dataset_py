__author__: "Ahmed Bahnasy"
# !mkdir "../output"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
use_gpu = torch.cuda.is_available()
use_gpu
# get the data from the csv file
df_train_path = "/kaggle/input/digit-recognizer/train.csv"
df_test_path = "/kaggle/input/digit-recognizer/test.csv"
X_train = pd.read_csv(df_train_path)
X_test = pd.read_csv(df_test_path)
# separate into two variables, features and labels
y_train = X_train['label']
X_train = X_train.drop('label', axis =1)
sns.set_style("darkgrid")
plt.figure(figsize=(12,8))
sns.countplot(y_train)
y_train.unique()
# show sample id
sample_id = 50
plt.imshow(X_train.loc[sample_id].values.reshape(28,28))
# plt.imshow(X_train.loc[4].reshape(28,28))
plt.title(str(y_train[sample_id]))
plt.show()
# normailize the data
X_train /= 255.0
X_test /= 255.0

# from df to numpy
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values

# split into test and validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)

# move to tensor
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
y_val = torch.from_numpy(y_val).type(torch.LongTensor)
# batch_size, epoch and iteration
batch_size = 100
num_epochs = 20

# create pytorch loaders
train = torch.utils.data.TensorDataset(X_train,y_train)
val = torch.utils.data.TensorDataset(X_val,y_val)
# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
val_loader = DataLoader(val, batch_size = batch_size, shuffle = False)




class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        # Network architecture
        self.conv1 = nn.Conv2d(1,6,(5,5), padding=2)
        self.conv2 = nn.Conv2d(6,16,(5,5))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))
        
        # flatten the input
        shape = x.size()[1:]
        features = 1
        for s in shape:
            features *= s
        x = x.view(-1, features)
        
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x
# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2 
        out = self.maxpool2(out)
        # flatten
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        return out
# Lenet with dropout

class LeNet_dropout(nn.Module):
    def __init__(self):
        super(LeNet_dropout, self).__init__()
        self.conv1 = nn.Conv2d(1,6,(5,5), padding=2)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(6,16,(5,5))
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))
        x = self.dropout2(x)
        
        # flatten the input
        shape = x.size()[1:]
        features = 1
        for s in shape:
            features *= s
        x = x.view(-1, features)
        
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x
# lenet SGD wit momentum
model = LeNet()
if use_gpu:
	model = model.cuda()
model = LeNet_dropout()
if use_gpu:
	model = net.cuda()
model = CNNModel()
if use_gpu:
	model = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
count = 0
loss_list = []
val_loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for images, labels in (train_loader):
        train_batch = Variable(images.view(batch_size, 1, 28, 28), requires_grad = True)
        labels_batch = Variable(labels, requires_grad = False)
        # move mini_batch to gpu
        if use_gpu:
            train_batch  = train_batch.cuda()
            labels_batch = labels_batch.cuda()
        # clear gradients
        optimizer.zero_grad()
        
        # forward propagation
        outputs = model(train_batch)
        
        # calculate loss
        loss = criterion(outputs, labels_batch)
        #backprobagation
        loss.backward()
        # update weights
        optimizer.step()
        
        count +=1
        
        # evaluate on the validation data every 50 iterations
        if count % 50 == 0:
            # calculate accuracy
            correct = 0
            total = 0
            for images, labels in val_loader:
                val_batch = Variable(images.view(-1,1,28,28), requires_grad = False)
                labels_batch = Variable(labels, requires_grad = False)
                if use_gpu:
                    val_batch  = val_batch.cuda()
                    labels_batch = labels_batch.cuda()
                outputs = model(val_batch)
                pred = torch.max(outputs.data, 1)[1]
                
                # calculate loss
                val_loss = criterion(outputs, labels_batch)
                
    
                total += len(labels)
    
                correct += (pred == labels_batch).sum()
                
            accuracy = (correct /float(total)) * 100

            loss_list.append(loss.data)
            val_loss_list.append(val_loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            
        if count % 500 == 0:
            print("Iteration: {}, Loss: {}, Accuracy: {}".format(count, loss.data, accuracy))
                
        
        
        # print insights every 500 iterations
        
# visualization loss 
plt.figure(figsize=(15,8))
plt.plot(iteration_list,loss_list, label="training")
plt.plot(iteration_list,val_loss_list, label='validation')
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.legend()
plt.show()


plt.figure(figsize=(15,8))
# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()
# create the output file container
output_file = np.ndarray(shape=(n_test_samples,2), dtype = int)

for test_idx in range(n_test_samples):
    test_sample = X_test[test_idx].clone().unsqueeze(dim = 1)
    test_sample = test_sample.type(torch.FloatTensor)
    if use_gpu:
        test_sample = test_sample.cuda()
    pred = net(test_sample)
    # get the index of the max class
    _ , pred = torch.max(pred,1)
    output_file[test_idx][0] = test_idx+1
    output_file[test_idx][1] = pred
    
    if test_idx % 1000 ==0:
        print(f"testing sample #{test_idx}")
        


submission = pd.DataFrame(output_file, dtype=int, columns=['ImageId', 'Label'])

    
    
# Sanity check
sample = 460
plt.imshow(X_test[sample].reshape(28,28))

print(output_file[sample][1])
print("Generating output file\n")
submission.to_csv('pytorch_LeNet.csv', index=False, header=True)
# from IPython.display import FileLink
# FileLink(r'pytorch_LeNet.csv')
submission.shape