# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pathlib import Path

import os

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import numpy as np

import PIL

import torch

import torch.nn as nn

import torch.optim as optim

import torchvision.datasets as dset



from torchvision import transforms

import torch

from torchvision import models

from torch.utils.data import Dataset

from torch.utils.data.sampler import SubsetRandomSampler



from torchvision import transforms



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



device = torch.device("cuda:0") # Let's make sure GPU is available!
!ls ../input/chest-xray-pneumonia/chest_xray/chest_xray/train/
from skimage import io, transform

from PIL import Image



class PictureReader(Dataset):

    TYPES = ["NORMAL", "PNEUMONIA"]

    def __init__(self, folder, transform=None):

        self.transform = transform

        self.folder = folder

        self.files = []

        for t in self.TYPES:

            tmp_folder = os.path.join(folder, t)

            for file in os.listdir(tmp_folder):

                if "jpg" in file or "jpeg" in file:

                    self.files.append((os.path.join(tmp_folder, file), t))

        # TODO: Your code here!

        

    def __len__(self):

        return len(self.files)

    

    def __getitem__(self, index):        

        img_id, y = self.files[index]

        y = int(y == "PNEUMONIA")

        try:

            img = io.imread(img_id)

            if self.transform:

                tmp = Image.fromarray(img)

                if tmp.getbands()[0] == 'L':

                    tmp = tmp.convert('RGB')

                tmp = self.transform(tmp)

                img = np.array(tmp)

        except:

            print(img_id)

            return 0/0

        return img, y, img_id
def visualize_samples(dataset, indices, title=None, count=10):

    # visualize random 10 samples

    plt.figure(figsize=(count*3,3))

    display_indices = indices[:count]

    if title:

        plt.suptitle("%s %s/%s" % (title, len(display_indices), len(indices)))        

    for i, index in enumerate(display_indices):    

        x, y, _ = dataset[index]

        plt.subplot(1,count,i+1)

        plt.title("Label: %s" % y)

        plt.imshow(x)

        plt.grid(False)

        plt.axis('off')   

    

orig_dataset = PictureReader(

    "../input/chest-xray-pneumonia/chest_xray/chest_xray/train/",

    transform=transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ColorJitter(hue=.20, saturation=.20),

    ]))

indices = np.random.choice(np.arange(len(orig_dataset)), 7, replace=False)



visualize_samples(orig_dataset, indices, "Samples")
train_dataset = PictureReader("../input/chest-xray-pneumonia/chest_xray/chest_xray/train/", 

                       transform=transforms.Compose([

                           transforms.Resize((256, 256)),

                           transforms.ToTensor(),

                           transforms.Normalize(mean=[0.485],

                                 std=[0.229])                         

                       ]))

test_dataset = PictureReader("../input/chest-xray-pneumonia/chest_xray/chest_xray/test", 

                       transform=transforms.Compose([

                           transforms.Resize((256, 256)),

                           transforms.ToTensor(),

                           transforms.Normalize(mean=[0.485],

                                 std=[0.229])                         

                       ]))

val_dataset = PictureReader("../input/chest-xray-pneumonia/chest_xray/chest_xray/val", 

                       transform=transforms.Compose([

                           transforms.Resize((256, 256)),

                           transforms.ToTensor(),

                           transforms.Normalize(mean=[0.485],

                                 std=[0.229])                         

                       ]))
batch_size = 64



train_indices, val_indices = list(range(len(train_dataset))), list(range(len(val_dataset)))



np.random.seed(42)

np.random.shuffle(train_indices)



train_sampler = SubsetRandomSampler(train_indices)

val_sampler = SubsetRandomSampler(val_indices)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

# Notice that we create test data loader in a different way. We don't have the labels.

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):    

    loss_history = []

    train_history = []

    val_history = []

    for epoch in range(num_epochs):

        model.train() # Enter train mode

        

        loss_accum = 0

        correct_samples = 0

        total_samples = 0

        for i_step, (x, y, z) in enumerate(train_loader):

            x_gpu = x.to(device)

            y_gpu = y.to(device)

            prediction = model(x_gpu)    

            loss_value = loss(prediction, y_gpu)

            optimizer.zero_grad()

            loss_value.backward()

            optimizer.step()



            _, indices = torch.max(prediction, 1)

            correct_samples += torch.sum(indices == y_gpu)

            total_samples += y.shape[0]



            loss_accum += loss_value



        ave_loss = loss_accum / i_step

        train_accuracy = float(correct_samples) / total_samples

        val_accuracy = compute_accuracy(model, val_loader)

        

        loss_history.append(float(ave_loss))

        train_history.append(train_accuracy)

        val_history.append(val_accuracy)

        

        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

        

    return loss_history, train_history, val_history

        

def compute_accuracy(model, loader):

    """

    Computes accuracy on the dataset wrapped in a loader

    

    Returns: accuracy as a float value between 0 and 1

    """

    model.eval() # Evaluation mode

    # TODO: Copy implementation from previous assignment

    # Don't forget to move the data to device before running it through the model!

    correct_samples = 0

    total_samples = 0

    for x, y,_  in loader:

        x_gpu = x.to(device)

        y_gpu = y.to(device)

        prediction = model(x_gpu)

        _, indices = torch.max(prediction, 1)

        correct_samples += torch.sum(indices == y_gpu)

        total_samples += y.shape[0]

    return float(correct_samples) / total_samples
class Flattener(nn.Module):

    def forward(self, x):

        batch_size, *_ = x.shape

        return x.view(batch_size, -1)
nn_model = nn.Sequential(

            nn.Conv2d(3, 16, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(4),

            nn.Conv2d(16, 64, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(4), 

            nn.Conv2d(64, 256, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(4), 

            Flattener(),

            nn.Linear(256*4*4, 2),

          )



nn_model.type(torch.cuda.FloatTensor)

#nn_model.type(torch.FloatTensor)

nn_model.to(device)



loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)

#loss = nn.CrossEntropyLoss().type(torch.FloatTensor)

optimizer = optim.SGD(nn_model.parameters(), lr=1e-1, weight_decay=1e-4)
#loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, 6)
device 
nn_model = nn.Sequential(

            nn.Conv2d(3, 8, 7),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(5),

            nn.Conv2d(8, 64, 5, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(4), 

            nn.Conv2d(64, 128, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2), 

            Flattener(),

            #torch.nn.BatchNorm1d(128*6*6),

            nn.Linear(128*6*6, 1000),

            #torch.nn.BatchNorm1d(1000),

            nn.ReLU(inplace=True),

            nn.Linear(1000, 2)

          )



nn_model.type(torch.cuda.FloatTensor)

nn_model.to(device)



loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)

optimizer = optim.SGD(nn_model.parameters(), lr=1e-2, weight_decay=1e-2)



loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, 6)
nn_model = nn.Sequential(

            nn.Conv2d(3, 8, 7),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(5),

            nn.Conv2d(8, 64, 5, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(4), 

            nn.Conv2d(64, 128, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2), 

            Flattener(),

            torch.nn.BatchNorm1d(128*6*6),

            nn.Linear(128*6*6, 1000),

            torch.nn.BatchNorm1d(1000),

            nn.ReLU(inplace=True),

            nn.Linear(1000, 2)

          )



nn_model.type(torch.cuda.FloatTensor)

nn_model.to(device)



loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)

optimizer = optim.SGD(nn_model.parameters(), lr=1e-2, weight_decay=1e-2)



loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, 6)
loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, 5)
i = 0.1

j = 0.1

while i > 0.0001:

    j = 0.1

    while j > 0.0001:

        print("#"*50)

        print(i, j)

        optimizer = optim.SGD(nn_model.parameters(), lr=i, weight_decay=j)

        loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, 5)

        print("#"*50)

        j /= 10

    i /= 10
nn_model.eval()

test_count = 0

test_acc = 0

for x, y, z in test_loader:

    x_gpu = x.to(device)

    y_gpu = y.to(device)

    pred = nn_model(x_gpu)

    _, ind = torch.max(pred, 1)

    test_count += 1

    test_acc += torch.sum(ind == y_gpu)

acc = float(test_acc) / test_count

print(test_count, test_acc, float(acc))