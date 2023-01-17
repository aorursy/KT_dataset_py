import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')         
print(device)
train_data = np.genfromtxt('../input/digit-recognizer/train.csv', delimiter=',')
test_data = np.genfromtxt('../input/digit-recognizer/test.csv', delimiter=',')
train_images = train_data[1:, 1:]
train_labels = train_data[1:, 0]

test_images = test_data[1:]
class my_train_dataset():    

    def __init__(self, train_images, train_labels):

        super(my_train_dataset).__init__()
        
        self.X = train_images.reshape(-1,1,28,28)       
        
        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(train_labels).long()
        
            
    def __getitem__(self,index):
        
        image = self.X[index]
        label= self.Y[index]

        return image, label
        
    def __len__(self):
        return len(self.X)



class my_test_dataset():    

    def __init__(self, test_images):

        super(my_test_dataset).__init__()
        
        self.X = test_images.reshape(-1,1,28,28)       
        
        self.X = torch.from_numpy(self.X).float()
        
            
    def __getitem__(self,index):
        
        image = self.X[index]

        return image
        
    def __len__(self):
        return len(self.X)
#### Define CNN Model ####

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
      
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding = 2)
        self.hidden = nn.Linear(128,64)
        self.output = nn.Linear(64, 10)
        # self.dropout = nn.Dropout(0.3)

        
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))       
        x = F.relu(self.conv3(x)) 
        x = F.avg_pool2d(x, [x.size(2), x.size(3)], stride=1)
        x = x.reshape(x.shape[0],x.shape[1])
        x = self.hidden(x)
        # x = self.dropout(x)
        x = self.output(x)

        return x
net = Net()
net.to(device)
net
Training_Loss = []
Training_Accuracy = []

def train(model, data_loader, epochs):
    net.train()
    for epoch in range(epochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            outputs = model(feats)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss


        train_loss, train_acc = test_classify(model, data_loader)
        print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\t'.
              format(train_loss, train_acc))
        Training_Loss.append(train_loss)
        Training_Accuracy.append(train_acc)
    
    
    
def test_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, labels.long())

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total
#Training Batch size
Batch_size = 256

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)

# Epochs
num_Epochs = 25
##### Train Dataloader #### 
train_dataset = my_train_dataset(train_images,train_labels)          
train_dataloader = data.DataLoader(train_dataset, shuffle= True, batch_size = Batch_size, num_workers=4,pin_memory=True)


#### Test Dataloader ####
test_dataset = my_test_dataset(test_images)
test_dataloader = data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)
train(net, train_dataloader, epochs = num_Epochs)
import csv

def predict(model, test_loader):
    
    model.eval()
    total = 0
    index= 1
    with open('output.csv',mode='w') as output_file:
      
        f=csv.writer(output_file,delimiter=',')
        f.writerow(['ImageId','Label'])


        for batch_num, (feats) in enumerate(test_loader):
            feats = feats.to(device)
            outputs = model(feats)

            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)

            f.writerow([str(index), int(pred_labels)])
            index+=1

predict(net, test_dataloader)
plt.figure(figsize=(10,10))
x = np.arange(1,26)
plt.plot(x, Training_Loss, label = 'Training Loss')
plt.xlabel('Epochs', fontsize =16)
plt.ylabel('Loss', fontsize =16)
plt.title('Loss v/s Epochs',fontsize =16)
plt.legend(fontsize=16)

plt.figure(figsize=(10,10))
plt.plot(x, Training_Accuracy, label = 'Training Accuracy')
plt.xlabel('Epochs', fontsize =16)
plt.ylabel('Accuracy', fontsize =16)
plt.title('Accuracy v/s Epochs',fontsize =16)
plt.legend(fontsize=16)

