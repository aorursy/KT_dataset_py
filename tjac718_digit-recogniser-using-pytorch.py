#Using the GPU for faster computation
import sys
print(sys.version)
device = 'cuda'
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Checking for GPU availability
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.is_available())
"""
#Loading the training data
train = pd.read_csv("../input/digit-recognizer/train.csv")
"""
class DatasetMNIST(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomAffine(degrees = 2, translate = (0.02, 0.02), scale = (0.98, 1.02)), transforms.ToTensor()])
train_dataset = DatasetMNIST("../input/digit-recognizer/train.csv", transform)
train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)

train_transform_x = []
train_transform_y = []

for xb, yb in train_loader:
    train_transform_x.append(xb)
    train_transform_y.append(yb)

train_x = torch.cat(train_transform_x, dim=1)  
train_y = torch.cat(train_transform_y, dim=0) 
    
train_x = train_x.reshape(-1, 1, 28, 28)
train_y = train_y.reshape(-1, 1)
    
train_x = train_x.numpy()
train_y = train_y.numpy()
    
train_x = train_x/255.0
    
print(train_x.shape)
print(train_y.shape)
random_seed = 1

X_train, X_cv, Y_train, Y_cv = train_test_split(train_x, train_y, test_size = 0.2, random_state = random_seed)

#Just checking the values with a random index
X_train = X_train.reshape(-1, 28, 28)
plt.imshow(X_train[6], cmap = 'gray')
print(Y_train[6])
X_train = X_train.reshape(-1, 1, 28, 28)
#Just checking to see if the relevant shapes are as expected

print(X_train.shape)
print(Y_train.shape)
print(X_cv.shape)
print(Y_cv.shape)

#isinstance(X_train, np.ndarray)
#isinstance(test, np.ndarray)
#isinstance(Y_train, np.ndarray)
#Converting numpy arrays to torch tensors

X_train,Y_train, X_cv, Y_cv = map(torch.tensor, (X_train, Y_train, X_cv, Y_cv))
train_ds = TensorDataset(X_train, Y_train)
train_dl = DataLoader(train_ds, batch_size = 128, shuffle = True)

valid_ds = TensorDataset(X_cv, Y_cv)
valid_dl = DataLoader(valid_ds, batch_size = 256)

class Mnist_cnn(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 1, padding = 2)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 5, stride = 1, padding = 2)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv4_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(3136, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = self.conv1_bn(xb)
        xb = F.relu(self.conv2(xb))
        xb = self.conv2_bn(xb)
        xb = F.max_pool2d(xb, (2, 2))
        xb = self.conv2_drop(xb)
        xb = F.relu(self.conv3(xb))
        xb = self.conv3_bn(xb)
        xb = F.relu(self.conv4(xb))
        xb = self.conv4_bn(xb)
        xb = F.max_pool2d(xb, (2, 2), stride = 2)
        xb = self.conv4_drop(xb)
        xb = xb.view(-1, self.num_flat_features(xb))
        xb = F.relu(self.fc1(xb))
        xb = self.fc1_bn(xb)
        xb = self.fc1_drop(xb)
        xb = self.fc2(xb)
        
        return xb
  
    def num_flat_features(self, xb):
        size = xb.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
model = Mnist_cnn()
print(model) 
model = model.float()

#Passing the model to the GPU
model.to(device)
opt = optim.Adam(model.parameters(), lr = 0.001)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience = 1, verbose = True)

loss_func = nn.CrossEntropyLoss()
def fit(model, epochs, opt, loss_func, train_dl, valid_dl):

    for epoch in range(epochs):
        #Going into training mode
        model.train()
        
        train_loss = 0
        train_acc = 0
       
        for xb, yb in train_dl:
            xb = xb.to(device)   #Passing the input mini-batch to the GPU
            yb = yb.to(device)   #Passing the label mini-batch to the GPU
            opt.zero_grad()      #Setting the grads to zero to avoid accumulation of gradients
            out = model(xb.float())
            loss = loss_func(out, yb.squeeze())    #Squeezing yb so it has dimensions (minibatch_size,)
            train_loss += loss
            train_pred = torch.argmax(out, dim = 1)
            train_pred = train_pred.reshape(train_pred.size()[0], 1) #Setting train_pred to have shape (minibatch_size, 1)
            train_acc += (train_pred == yb).float().mean()
            
            loss.backward()
            opt.step()
        
        lr_scheduler.step(train_loss/len(train_dl))   #Setting up lr decay  
        
        model.eval()            #Going into eval mode                            
        with torch.no_grad():   #No backprop
            valid_loss = 0
            valid_acc = 0
            
            for xb, yb in valid_dl:
                xb = xb.to(device)  
                yb = yb.to(device)
                cv_out = model(xb.float())
                valid_loss += loss_func(cv_out, yb.squeeze())
                valid_pred = torch.argmax(cv_out, dim = 1)
                valid_pred = valid_pred.reshape(valid_pred.size()[0], 1)
                valid_acc += (valid_pred == yb).float().mean()
        
        print("Epoch ", epoch, " Training Loss: ", train_loss/len(train_dl), "CV Loss: ", valid_loss/len(valid_dl))
        print("Training Acc: ", train_acc/len(train_dl), "CV Acc: ", valid_acc/len(valid_dl))
fit(model, 200, opt, loss_func, train_dl, valid_dl)   #Training for 200 epoch
#Deleting some variables of no use later to free some space
del train_dl, valid_dl, train_ds, valid_ds, X_train, Y_train, X_cv, Y_cv
test = pd.read_csv("../input/digit-recognizer/test.csv")

test = test/255.0
test = test.values.reshape(-1, 1, 28, 28)
test = torch.from_numpy(test)
with torch.no_grad():
    test = test.to(device)    #Passing the entire test set to the GPU
    test_out = model(test.float())
    test_pred = torch.argmax(test_out, dim = 1)
    test_pred = test_pred.reshape(test_pred.size()[0], 1)
    test_pred_np = test_pred.cpu().numpy()   #Conversion of tensor to numpy array
test_pred_np = np.reshape(test_pred_np, test_pred_np.shape[0])  #Reshaping to expected 1D array
test_pred_pd = pd.Series(test_pred_np, name = "Label")          #Conversion to Pandas Dataframe
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), test_pred_pd], axis = 1)
submission.to_csv("cnn_mnist_datagen.csv", index=False)