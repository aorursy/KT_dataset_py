import numpy as np

import pandas as pd 

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break



# Any results you write to the current directory are saved as output.
import torch

from torch import nn

from torchvision import datasets, models, transforms

import torch.utils.data as tdata

print(os.listdir("../input"))
seed = 123456

np.random.seed(seed)

torch.manual_seed(seed)
transform_to_tensor = transforms.Compose([

    transforms.Resize((200,200)),

    transforms.ToTensor(),

])

data_path_format = '../input/intel-image-classification/seg_{0}/seg_{0}'
image_datasets = dict(zip(('dev', 'test'), [datasets.ImageFolder(data_path_format.format(key),transform=transform_to_tensor) for key in ['train', 'test']]))

print(image_datasets)
from sklearn import model_selection

devset_indices = np.arange(len(image_datasets['dev']))

devset_labels = image_datasets['dev'].targets

train_indices, val_indices, train_labels,  val_labels = model_selection.train_test_split(devset_indices, devset_labels, test_size=0.1, stratify=devset_labels)
image_datasets['train'] = tdata.Subset(image_datasets['dev'], train_indices)

image_datasets['validation'] = tdata.Subset(image_datasets['dev'], val_indices)
from matplotlib import pyplot as plt

image_dataloaders = {key: tdata.DataLoader(image_datasets[key], batch_size=16,shuffle=True) for key in  ['train', 'validation']}

image_dataloaders['test'] = tdata.DataLoader(image_datasets['test'], batch_size=32)

def imshow(inp, title=None, fig_size=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0)) # C x H x W  # H x W x C

    #inp = channel_stds * inp + channel_means

    inp = np.clip(inp, 0, 1)

    fig = plt.figure(figsize=fig_size)

    ax = fig.add_subplot('111')

    ax.imshow(inp)

    if title is not None:

        ax.set_title(title)

    ax.set_aspect('equal')

    plt.pause(0.001)  
imshow(image_datasets['train'][8555][0])
import torch.nn as nn

from sklearn import metrics

import torch.nn.functional as F

from torch.nn.functional import log_softmax

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv = nn.Sequential(         

        nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3), 

        nn.ReLU(),

        nn.Conv2d(in_channels=20, out_channels=30, kernel_size=4), 

        nn.ReLU(),

        nn.MaxPool2d(6),

        nn.Conv2d(in_channels=30, out_channels=30, kernel_size=4), 

        nn.ReLU(),

        nn.MaxPool2d(3),

        nn.Conv2d(in_channels=30, out_channels=30, kernel_size=4),

        nn.MaxPool2d(3),

        nn.ReLU(), 

        )

        self.classifier = nn.Sequential(

            nn.Linear(120,200),

            nn.ReLU(),

            nn.Linear(200,6)

        )

    def forward(self, x):

        h = self.conv(x)

        h = h.view(x.size(0), -1)

        return  self.classifier(h)

    
from sklearn.metrics import accuracy_score

def fit(net,crit,train_loader,val_loader,optimizer, epochs):

    best=0

    net.cuda()

    for i in range(epochs):

        tr_loss = 0

        val_loss = 0

        val_accuracy =0

        for xx,yy in train_loader:

            xx, yy = xx.cuda(), yy.cuda()

            optimizer.zero_grad()

            y = net.forward(xx)

            loss = crit(y,yy)

            tr_loss += loss

            loss.backward()

            optimizer.step()

        tr_loss /= len(train_loader)

        with torch.no_grad():

            for xx,yy in val_loader:

                all_preds = []

                xx, yy = xx.cuda(), yy.cuda()

                y = net.forward(xx)

                loss = crit(y,yy)

                val_loss += loss

                all_preds.extend(y.argmax(dim=1).cpu().numpy())

                yy = yy.cpu().numpy()

                val_accuracy += accuracy_score(all_preds,yy)

            val_accuracy /= len(val_loader)

            if val_accuracy>best:

                best = val_accuracy

                torch.save(net.state_dict(), "../model.py")

        print(("epoch:%d, train loss:%f, validation accuracy:%f" % (i,tr_loss.item(),val_accuracy.item())))

    net.cpu()

    print("Train ended. Best accuracy is %f" % float(best))
import gc

gc.collect()

model = Net()

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

criterion = nn.CrossEntropyLoss() 

fit(model,criterion,image_dataloaders['train'],image_dataloaders['validation'],optimizer,15)
all_preds = []

correct_preds = []

for xx, yy in image_dataloaders['test']:

    model.cuda()

    xx = xx.cuda()

    output = model.forward(xx)

    all_preds.extend(output.argmax(1).tolist())

    correct_preds.extend(yy.tolist())

model.cpu()

all_preds = np.asarray(all_preds)

correct_preds = np.asarray(correct_preds)

target_names = image_datasets['test'].classes

print(metrics.classification_report(correct_preds, all_preds,target_names=target_names))