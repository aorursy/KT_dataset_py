# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import torch

from torch import nn

from torchvision import datasets, models, transforms

import torch.utils.data as tdata

!ls ../input/seg_test/seg_test/
data_path_format = '../input/seg_{0}/seg_{0}'

np.random.seed(5315)

torch.manual_seed(9784)

channel_means = (0.485, 0.456, 0.406)

channel_stds = (0.229, 0.224, 0.225)



image_transforms = transforms.Compose([

    transforms.Resize((150,150)),

    transforms.ToTensor(),

    transforms.Normalize(channel_means, channel_stds)

])

    

image_datasets = dict(zip(('dev', 'test'), [datasets.ImageFolder(data_path_format.format(key),transform=image_transforms) for key in ['train', 'test']]))

print(image_datasets)
devset_indices = np.arange(len(image_datasets['dev']))

devset_labels = image_datasets['dev'].targets
from sklearn import model_selection

train_indices, val_indices, train_labels,  val_labels = model_selection.train_test_split(devset_indices, devset_labels, test_size=0.1, stratify=devset_labels)
image_datasets['train'] = tdata.Subset(image_datasets['dev'], train_indices)

image_datasets['validation'] = tdata.Subset(image_datasets['dev'], val_indices)
from IPython.display import display

import matplotlib.pyplot as plt

image_dataloaders = {key: tdata.DataLoader(image_datasets[key], batch_size=16,shuffle=True) for key in  ['train', 'validation']}

image_dataloaders['test'] = tdata.DataLoader(image_datasets['test'], batch_size=32)



def imshow(inp, title=None, fig_size=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0)) # C x H x W  # H x W x C

    inp = channel_stds * inp + channel_means

    inp = np.clip(inp, 0, 1)

    fig = plt.figure(figsize=fig_size)

    ax = fig.add_subplot('111')

    ax.imshow(inp)

    if title is not None:

        ax.set_title(title)

    ax.set_aspect('equal')

    plt.pause(0.01)  



imshow(image_datasets['train'][8555][0]) # 5946   
cuda_device = torch.device('cuda')

cpu_device = torch.device('cpu')

device = cuda_device
import torch.nn as nn

import math





class MODEL(nn.Module):

    def __init__(self, num_classes=6):

        super(MODEL, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),



            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=10),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),

            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.ReLU(),

            nn.AvgPool2d(kernel_size=10))

        

        self.classifier = nn.Sequential(

            # input shape: (batch_size, 3, 224, 224) and

            # downsampled by a factor of 2^5 = 32 (5 times maxpooling)

            # So features' shape is (batch_size, 7, 7, 512)

            nn.Linear(in_features=128, out_features=128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(in_features=128, out_features=num_classes)

        )



        # initialize parameters

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels

                module.weight.data.normal_(0, math.sqrt(2. / n))

                module.bias.data.zero_()

            elif isinstance(module, nn.Linear):

                module.weight.data.normal_(0, 0.01)

                module.bias.data.zero_()



    def forward(self, x):

        x = self.features(x)

        # flatten

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x
def train_model(epochs, model, optimizer, criterion, loaders, n_prints=1):

    best_accuracy=0

    print_every = len(loaders['train']) // n_prints

    model.cuda()

    for epoch in range(epochs):

        model.train()

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

            

            if accuracy > best_accuracy:

                best_accuracy = accuracy

                torch.save(model.state_dict(), "../model.csv")

            

            print(f"Epoch {epoch}, val_loss {mean_val_loss}, accuracy = {accuracy}")
model = MODEL()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

criterion = nn.CrossEntropyLoss()

train_model(20, model, optimizer, criterion,image_dataloaders)
from sklearn.metrics import classification_report



model.eval()



all_preds = []

correct_preds = []

with torch.no_grad():

    for xx, yy in image_dataloaders['test']:

        xx = xx.to(device)

        output = model.forward(xx)

        all_preds.extend(output.argmax(1).tolist())

        correct_preds.extend(yy.tolist())

print(classification_report(all_preds,correct_preds))





model.load_state_dict(torch.load("../model.csv"))

all_preds = []

correct_preds = []

with torch.no_grad():

    for xx, yy in image_dataloaders['test']:

        xx = xx.to(device)

        output = model.forward(xx)

        all_preds.extend(output.argmax(1).tolist())

        correct_preds.extend(yy.tolist())

print(classification_report(all_preds,correct_preds))
from sklearn import metrics

target_names = image_datasets['test'].classes

all_preds = np.asarray(all_preds)

correct_preds = np.asarray(correct_preds)



confusion_matrix = metrics.confusion_matrix(correct_preds, all_preds)

pd.DataFrame(confusion_matrix, index=target_names, columns=target_names)