# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os





import torch

from torch import nn

from torchvision import datasets, models, transforms

import torch.utils.data as tdata



np.random.seed(5315)

torch.manual_seed(9784)

print(os.listdir("../input"))
channel_means = (0.485, 0.456, 0.406)

channel_stds = (0.229, 0.224, 0.225)



image_transforms = transforms.Compose([

    transforms.Resize((150,150)),

    transforms.ToTensor(),

    transforms.Normalize(channel_means, channel_stds)

])

data_path_format = '../input/seg_{0}/seg_{0}'

image_datasets = dict(zip(('dev', 'test'), [datasets.ImageFolder(data_path_format.format(key),transform=image_transforms) for key in ['train', 'test']]))

print(image_datasets)
devset_indices = np.arange(len(image_datasets['dev']))

devset_labels = image_datasets['dev'].targets



from sklearn import model_selection

train_indices, val_indices, train_labels,  val_labels = model_selection.train_test_split(devset_indices, devset_labels, test_size=0.1, stratify=devset_labels)
image_datasets['train'] = tdata.Subset(image_datasets['dev'], train_indices)

image_datasets['validation'] = tdata.Subset(image_datasets['dev'], val_indices)
image_dataloaders = {key: tdata.DataLoader(image_datasets[key], batch_size=16,shuffle=True) for key in  ['train', 'validation']}

image_dataloaders['test'] = tdata.DataLoader(image_datasets['test'], batch_size=32)

cuda_device = torch.device('cuda')

cpu_device = torch.device('cpu')

device = cuda_device
import torch.nn as nn

import torch.nn.functional as F



class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),

            

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),

            

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.MaxPool2d(5))

            

        self.classifier = nn.Sequential(

            nn.Linear(6272,5000),nn.ReLU(),

            nn.Linear(5000,100), nn.ReLU(),

            nn.Linear(100,6),

        )

    def forward(self, x):

        h = self.conv(x)

        h = h.view(x.size(0), -1)

        return  F.log_softmax(self.classifier(h))
def train_model(epochs, model, optimizer, criterion, loaders, device, n_prints=1):

    print_every = len(loaders['train']) // n_prints

    for epoch in range(epochs):

        best_acc = 0

        model.train()

        model.cuda()

        running_train_loss = 0.0

        

        for iteration, (xx, yy) in enumerate(loaders['train']):

            optimizer.zero_grad()

            xx, yy = xx.to(device), yy.to(device)

            out = model(xx)

            loss = criterion(out, yy)

            running_train_loss += loss.item()

            loss.backward()

            optimizer.step()

            

            if(iteration % print_every == print_every - 1):

                running_train_loss /= print_every

                print(f"Epoch {epoch}, iteration {iteration} training_loss {running_train_loss}")

                running_train_loss = 0.0

            

        with torch.no_grad():

            model.eval()

            running_corrects = 0

            running_total = 0

            running_loss = 0.0

            for xx, yy in loaders['validation']:

                batch_size = xx.size(0)

                xx, yy = xx.to(device), yy.to(device)



                out = model(xx)

                

                loss = criterion(out, yy)

                running_loss += loss.item()

                

                predictions = out.argmax(1)

                running_corrects += (predictions == yy).sum().item()

                running_total += batch_size

            

            mean_val_loss = running_loss / len(loaders['validation'])

            accuracy = running_corrects / running_total

            

            if accuracy > best_acc:

                best_acc = accuracy

                torch.save(model.state_dict(), "../output/best_model.pytorch")

            

            print(f"Epoch {epoch}, val_loss {mean_val_loss}, accuracy = {accuracy}")

            

    model.load_state_dict(torch.load("../output/best_model.pytorch"))

                
os.mkdir("../output/")

model = Network()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

criterion = nn.CrossEntropyLoss()





train_model(20, model, optimizer, criterion,image_dataloaders, device, n_prints=1)
from sklearn.metrics import classification_report

all_preds = []

correct_preds = []

for xx,yy in image_dataloaders['test']:

    xx = xx.to(device)

    y_pred = model.forward(xx)

    all_preds.extend(y_pred.argmax(1).tolist())

    correct_preds.extend(yy.tolist())

    

print(classification_report(all_preds,correct_preds))