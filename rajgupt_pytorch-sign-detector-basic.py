import numpy as np
import pandas as pd
import sys, os
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# debugger for handling errors
from IPython.core.debugger import set_trace
# Custom Dataset
# Ref. https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
class SIGNSDataset(Dataset):
    def __init__(self, filenames, labels, transform):      
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        #return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        #open image, apply transforms and return with label
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]
# Once experimentation is done is the preprocessing to be done, transformation and preprocessing should be done as part of data loader

train_transformer = transforms.Compose([
                    transforms.Resize(64),              # resize the image to 64x64 
                    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
                    transforms.ToTensor()])             # transform it into a PyTorch Tensor

val_transformer =   transforms.Compose([
                    transforms.Resize(64),              # resize the image to 64x64 
                    transforms.ToTensor()])             # transform it into a PyTorch Tensor
import glob
images = glob.glob('../input/leapgestrecog/leapGestRecog/**/**/*.png')

# extract label number from filename and reduce by 1 so that it ranges from 0 to 9 (instead of 1 to 10). Otherwise loss function will complain
labels = [int(os.path.basename(i).split('_')[2])-1 for i in images]
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

train_dataset = SIGNSDataset(x_train, y_train, train_transformer)
val_dataset = SIGNSDataset(x_val, y_val, val_transformer)
bs = 32
lr = 0.01
train_dl = DataLoader(train_dataset, bs, shuffle=True)
val_dl = DataLoader(val_dataset, bs)
# Model Arch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(79680, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
import numpy as np

def get_model():
    model = Net()
    return model, optim.SGD(model.parameters(), lr=lr)


def loss_batch(model, loss_func, xb, yb, opt=None):
    yhat = model(xb)
    loss = loss_func(yhat, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

model, opt = get_model()
loss_func = F.cross_entropy

fit(1, model, loss_func, opt, train_dl, val_dl)