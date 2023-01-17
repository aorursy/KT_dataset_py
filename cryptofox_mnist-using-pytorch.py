# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
from skimage.transform import rotate



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Using GPU
print(sys.version)
device = 'cuda'
#Checking for GPU
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.is_available())

transform1 = transforms.Compose([transforms.ToPILImage(), transforms.RandomAffine(degrees = 30, translate = (0.1, 0.1), scale = (0.98, 1.02)), transforms.ToTensor()])
transform2 = transforms.Compose([transforms.ToPILImage(), transforms.RandomAffine(degrees = 15, translate = (0.15, 0.15), scale = (0.95, 1.05)), transforms.ToTensor()])

df = pd.read_csv('../input/digit-recognizer/train.csv')
train = torch.from_numpy(df.values)
x_train = (train[:, 1:])#.to(device)
y_train = (train[:, 0])#.to(device)
del train, df


df = pd.read_csv("../input/digit-recognizer/test.csv")
x_test = torch.from_numpy(df.values).to(device)
print(x_test.shape, y_train.shape, x_train.shape, y_train.max())

n, c = x_train.shape #n=no. of training egs & c=n_H *n_W for each eg
print(n, c)
print(x_train.shape, y_train.shape)


x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size = 0.1, random_state = random_seed)
n_train, _ = x_train.shape
n_cv, _ = x_cv.shape
print(n_train, n_cv)
x_train#.to(device)
y_train#.to(device)
x_cv#.to(device)
y_cv#.to(device)
print(x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_train.dtype, y_train.dtype)
'''x_train_temp = x_train.cpu().numpy().astype(np.float32)
print(x_train_temp.shape, x_train_temp.dtype)

final_train_x = []
final_train_y = []
for i in range(x_train.shape[0]):
    final_train_x.append(x_train_temp[i])
    final_train_x.append(rotate(x_train_temp[i].reshape((28, 28)), angle=15, mode = 'wrap').reshape((784)))
    final_train_x.append(rotate(x_train_temp[i].reshape((28, 28)), angle=8, mode = 'wrap').reshape((784)))
    final_train_x.append(rotate(x_train_temp[i].reshape((28, 28)), angle=-15, mode = 'wrap').reshape((784)))
    final_train_x.append(rotate(x_train_temp[i].reshape((28, 28)), angle=-8, mode = 'wrap').reshape((784)))
    for j in range(5):
        final_train_y.append(y_train[i].cpu().numpy().reshape(1))'''

#final_train_x = np.reshape(final_train_x, (len(final_train_x.shape), 28*28))
#final_train_y = np.reshape(final_train_y, (-1))

'''print(final_train_x[0].dtype, final_train_y[0].dtype)

random_seed = 1
x_train = torch.from_numpy(np.array(final_train_x)).to(device)
y_train = torch.from_numpy(np.array(final_train_y)).squeeze().to(device)

print(x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_train.dtype, y_train.dtype)'''
'''transformed_images =[]
transformed_labels = []
x_train_trans = 0
y_train_trans = 0


for i in range(n_train):
    orig = x_train[i].cpu().numpy().astype(np.uint8).reshape((28, 28))
    transformed1 = transform1(orig)
    transformed2 = transform2(orig)
    transformed_images.append(transformed1)
    transformed_images.append(transformed2)
    transformed_labels.append(y_train[i].reshape((-1, 1)))
    transformed_labels.append(y_train[i].reshape((-1, 1)))
    
print(transformed_images[1].shape, transformed_labels[1].shape)


x_train_trans = torch.cat(transformed_images, dim=0)#.to(device)
y_train_trans = torch.cat(transformed_labels, dim=0)#.to(device)
print(x_train_trans.shape, y_train_trans.shape)


x_train = torch.cat((x_train_trans, x_train), dim=0)#.to(device)
y_train = torch.cat((y_train_trans, y_train.reshape((-1, 1))), dim=0)#.to(device)
y_train = y_train.squeeze()
print(x_train.shape, y_train.shape)

'''

'''index = 3
tester = x_train[index].cpu().numpy().astype(np.uint8).reshape((28, 28))
tester3 = transformed_images[index].cpu().numpy().astype(np.uint8).reshape((28,28))


print (y_train[index])
tester1 = transform1(tester)
tester2 = transform2(tester)

plt.imshow(tester2.reshape((28, 28)), cmap='gray')'''

'''tester = x_test[7].cpu().numpy().astype(np.uint8).reshape((28, 28, 1))

plt.imshow(tester.reshape((28, 28)), cmap='gray')

print (tester.shape)'''


index = 5
plt.imshow(x_train[index].cpu().reshape((28, 28)), cmap='gray')
print (y_train[index])
index = 5
plt.imshow(x_cv[index].cpu().reshape((28, 28)), cmap='gray')
print (y_cv[index])

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_cv, y_cv)

def get_data(train_ds, bs):
    return DataLoader(train_ds, batch_size =bs, shuffle=True)

def preprocess(x, y):
  x = x/255.0
  return x.reshape(-1, 1, 28, 28).to(device), y.to(device)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


class Mnist_CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) #n_C_prev=1(no RGB Grayscale) n_C=16(#filters)
    self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2) #n_C_prev=16 n_C=16
    self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1) #n_C_prev=16 n_C=10
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) #n_C_prev=16 n_C=10
    self.lin1 = nn.Linear(3136, 256)
    self.lin2 = nn.Linear(256, 10)
    self.norm1 = nn.BatchNorm2d(32)
    self.norm2 = nn.BatchNorm2d(48)
    self.norm3 = nn.BatchNorm2d(64)
    self.drop1 = nn.Dropout2d(p=0.4)
    self.drop2 = nn.Dropout2d(p=0.4)
    self.norm4 = nn.BatchNorm1d(256)
    self.drop3 = nn.Dropout(p=0.4)
    #self.lin2 = nn.Linear(128, 10)
    


  def forward(self, xb):
    xb = xb.view(-1, 1, 28, 28)
    #1
    xb = F.relu(self.conv1(xb))
    xb = self.norm1(xb)
    #xb = F.avg_pool2d(xb, 2, 2)
    
    #2
    xb = F.relu(self.conv2(xb))
    xb = self.norm2(xb)
    xb = F.avg_pool2d(xb, 2, 2)
    xb = self.drop2(xb)
    
    #3
    xb = F.relu(self.conv3(xb))
    xb = self.norm3(xb)
    
    #4
    xb = F.relu(self.conv4(xb))
    xb = self.norm3(xb)
    xb = F.avg_pool2d(xb, 2, 2)
    xb = self.drop2(xb)
    
    #fc1
    xb = torch.flatten(xb, 1, 3)
    xb = F.relu(self.lin1(xb))
    xb = self.norm4(xb)
    xb = self.drop3(xb)
    
    #fc2
    xb = self.lin2(xb)
    return xb.view(-1, xb.size(1))

loss_func = F.cross_entropy 
model = Mnist_CNN()
model.float()
model.to(device)
def accuracy(out, yb):
    pred = torch.argmax(out, keepdim= False, dim=1)
    return (pred == yb).float().mean() 
'''for using negative log likelihood loss and log softmax activation,
Pytorch provides a single function F.cross_entropy that combines the two'''
def fit(model, epochs, train_dl, valid_dl, opt):
    losses = []
    
    for epoch in range(epochs):
        train_loss =0
        train_acc =0
        model.train()
        for xb, yb in train_dl:
            xb.to(device)
            yb.to(device)
            pred = model(xb.float())
            pred.to(device)
            train_acc += accuracy(pred, yb)
            loss = loss_func(pred, yb)
            train_loss += loss
            #back propogation
            loss.backward()
            opt.step()
            opt.zero_grad()      

        losses.append(train_loss)
        print("Iteration no: "+ str(epoch), "loss = "+str(losses[epoch].item()))
        print("Accuracy of train set:", train_acc/len(train_dl))
        lr_scheduler.step(train_loss/len(train_dl))
        
        
        model.eval()    
        with torch.no_grad():
            valid_acc=0
            loss_valid = 0
            for xb_valid, yb_valid in valid_dl:
                xb_valid.to(device)
                yb_valid.to(device)
                pred_valid = model(xb_valid.float())
                pred_valid.to(device)
                valid_acc += accuracy(pred_valid, yb_valid)
                loss_valid += loss_func(pred_valid, yb_valid)
            print("Accuracy of validation set :", valid_acc/len(valid_dl))

    plt.plot(losses)
    plt.ylabel("loss")
    plt.xlabel("iterations (per hundreds)")
    plt.show()
train_dl = get_data(train_ds, bs=64)
valid_dl = get_data(valid_ds, bs = 128)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

opt = optim.Adam(model.parameters(), lr=0.1)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1, verbose=True, eps = 10e-10)

fit(model, 160, train_dl, valid_dl, opt)


#Deleting some variables of no use later to free some space
del train_dl, valid_dl, x_train, y_train, x_cv, y_cv
x_test = x_test/255.0
x_test = x_test.reshape(-1, 1, 28, 28)
with torch.no_grad():
    x_test = x_test    #Passing the entire test set to the GPU
    test_out = model(x_test.float()).cpu()
    test_pred = torch.argmax(test_out, dim = 1, keepdim=True).cpu().numpy()
    #test_pred_np = test_pred.cpu().numpy() 
    test_pred = np.reshape(test_pred, test_pred.shape[0])
    print(test_pred.shape)
    
    row1 = pd.Series(test_pred, name="Label")
    row2 = pd.Series(range(1, 28001), name="ImageId")
    submission = pd.concat([row2, row1], axis=1)
    submission.to_csv("cnn_mnist_datagen.csv", index=False)
def print_accuracy(model):
    print("Accuracy of train set:", accuracy(model(x_train.float()), y_train))
    print("Accuracy of test set:", accuracy(model(x_cv.float()), y_cv))

print_accuracy(model)


