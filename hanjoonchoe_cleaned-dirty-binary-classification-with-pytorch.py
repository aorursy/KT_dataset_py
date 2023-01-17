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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import torch

import cv2

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

import torchvision

from PIL import Image
!rm -r plates
#Unzip jpg files

!unzip -o /kaggle/input/platesv2/plates.zip
cwd = os.getcwd()

path = os.path.join(cwd,'plates')

print(path)
!mkdir -p plates/valid/cleaned

!mkdir -p plates/valid/dirty
!shuf -n 10 -e /kaggle/working/plates/train/cleaned/* | xargs -i mv {} /kaggle/working/plates/valid/cleaned

!shuf -n 10 -e /kaggle/working/plates/train/dirty/* | xargs -i mv {} /kaggle/working/plates/valid/dirty
print(os.listdir(os.path.join(cwd,'plates','valid','cleaned')))

print(os.listdir(os.path.join(cwd,'plates','valid','dirty')))
transforms1 = transforms.Compose([

    torchvision.transforms.RandomPerspective(distortion_scale=0.3, p=0.3, interpolation=3),

    transforms.RandomRotation(20, resample=False, expand=False, center=None),

    transforms.CenterCrop(224),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize((0, 0, 0), (1, 1, 1)),

])

class Dataloader(Dataset):

    

    def __init__(self,root_dir,transforms=None):

        

        self.root_dir = root_dir

        self.classes = self._find_classes(self.root_dir)

        self.samples = self.make_samples(self.root_dir,self.classes)

        self.transforms = transforms

    def __len__(self):

        return len(self.samples)

    

    def __getitem__(self,idx):

        

        if ('train' in self.root_dir)|('valid' in self.root_dir):

            

            image = cv2.imread(self.samples[idx][0])

            image = cv2.resize(image,(224,224))

            target = self.samples[idx][1]

            if self.transforms:

                image = Image.fromarray(image)

                image = self.transforms(image)

            

            

            return image,target

        

        else:

            image = cv2.imread(self.samples[idx])

            image = cv2.resize(image,(224,224))

            return image

        

    

    def _find_classes(self,dir):

        

        if ('train' in dir)|('valid' in dir):

            classes = [d.name for d in os.scandir(dir) if d.is_dir()]

            classes_to_idx = {classes[i]: i for i in range(len(classes))}

            classes_to_idx['dirty'] = 1

            classes_to_idx['cleaned'] = 0

            return classes_to_idx

        else:

            return None



    def make_samples(self,dir,classes_to_idx):

        images = []

        if classes_to_idx != None:

            for target in sorted(classes_to_idx.keys()):

        

                d = os.path.join(dir,target)

            

                for root, _, fnames in sorted(os.walk(d,followlinks=True)):

                    for fname in sorted(fnames):

                        if '.jpg' in fname:

                            path = os.path.join(root,fname)

                            item = (path, classes_to_idx[target])

                            images.append(item)

        else:

            

            d = dir

            

            for root, _, fnames in sorted(os.walk(d,followlinks=True)):

                for fname in sorted(fnames):

                    if '.jpg' in fname:

                            path = os.path.join(root,fname)

                            item = path

                            images.append(item)

            

        return images

        
os.listdir(path)
train_path = os.path.join(path,'train')

train_dataset = Dataloader(train_path)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
a = next(iter(train_dataloader))

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 6, figsize=(25, 25))

for i in range(6):

    axs[i].imshow(a[0][i].numpy())

    if a[1][i].numpy() == 1:

        axs[i].set_title('Dirty')

    else:

        axs[i].set_title('Cleaned')
train_dataset = Dataloader(train_path,transforms=transforms1)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)

a = next(iter(train_dataloader))

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 6, figsize=(25, 25))

for i in range(6):

    axs[i].imshow(torch.transpose(a[0][i],0,2).numpy())

    if a[1][i].numpy() == 1:

        axs[i].set_title('Dirty')

    else:

        axs[i].set_title('Cleaned')
train_path = os.path.join(path,'train')

train_dataset = Dataloader(train_path,transforms=transforms1)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0)

valid_path = os.path.join(path,'valid')

valid_dataset = Dataloader(valid_path,transforms=transforms1)

valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=3, shuffle=True, num_workers=0)
def train_model(model, loss, optimizer, scheduler, num_epochs):

    global config

    train_losses =  []

    train_accs =  []

    val_losses =  []

    val_accs =  []

  # Set model to training mode

    #model.train()

    for epoch in tqdm(range(num_epochs)):

        running_loss = 0.

        running_acc = 0.

        print('Epoch {}/{}:'.format(epoch, num_epochs), flush=True)



        scheduler.step()

        # Iterate over data.

        for idx,(inputs, labels) in enumerate(train_dataloader):

            inputs = inputs.to(device)

            labels = labels.to(device)



            optimizer.zero_grad()



            #preds = model(torch.transpose(inputs, 1, 3).float())

            preds = model(inputs)#.transpose(inputs, 1, 3).float())

            loss_value = loss(preds, labels)

            preds_class = preds.argmax(dim=1)



            loss_value.backward()

            optimizer.step()



            # statistics

            running_loss += loss_value.item()

            running_acc += (preds_class == labels.data).float().mean()

            

        train_loss = running_loss / len(train_dataloader)

        train_acc = running_acc / len(train_dataloader)

        train_losses.append(train_loss)

        train_accs.append(train_acc)



        with torch.no_grad():

            #model.eval()

            running_loss = 0.

            running_acc = 0.

            for idx,(inputs,labels) in enumerate(valid_dataloader):

                inputs = inputs.to(device)

                labels = labels.to(device)

                #preds = model(torch.transpose(inputs, 1, 3).float())

                preds = model(inputs)#.transpose(inputs, 1, 3).float())

                loss_value = loss(preds, labels)

                preds_class = preds.argmax(dim=1)

                

                running_loss += loss_value.item()

                running_acc += (preds_class == labels.data).float().mean()



        val_loss = running_loss / len(valid_dataloader)

        val_acc = running_acc / len(valid_dataloader)

        val_losses.append(val_loss)

        val_accs.append(val_acc)

        print('train Loss: {:.4f} train Acc: {:.4f} val Loss: {:.4f} val Acc: {:.4f}'.format( train_loss, train_acc,val_loss,val_acc), flush=True)

    return model, train_losses,val_losses ,train_accs, val_accs
import torch.nn.functional as F
from torchvision import transforms, models

class PlatesNet(torch.nn.Module):

    def __init__(self):

        super(PlatesNet, self).__init__()

        self.resnet = models.resnet152(pretrained=True, progress=False)



        # Disable grad for all conv layers

        for param in self.resnet.parameters():

            param.requires_grad = False



        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features // 2)

        self.act = torch.nn.Sigmoid()

        self.fc = torch.nn.Linear(self.resnet.fc.in_features // 2, self.resnet.fc.in_features // 4)

        self.act1 = torch.nn.Sigmoid()

        self.fc1 = torch.nn.Linear(self.resnet.fc.in_features // 4, 2)

    

    def forward(self, X):

        X = self.resnet(X)

        X = self.act(X)

        X = self.fc(X)

        X = self.act1(X)

        X = self.fc1(X)

        

        return X
model = PlatesNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)



loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
from tqdm import tqdm_notebook as tqdm
model, train_loss,val_loss, train_acc, val_acc = train_model(model, loss, optimizer, scheduler, num_epochs=200)
#import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(15, 5))

axs[0,0].plot(train_acc)

axs[0,0].set_title('Train Accuracy')

axs[0,1].plot(train_loss)

axs[0,1].set_title('Train Loss')

axs[1,0].plot(val_acc)

axs[1,0].set_title('Val Accuracy')

axs[1,1].plot(val_loss)

axs[1,1].set_title('Val Loss')
test_path = os.path.join(path,'test')

test_dataset = Dataloader(test_path)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=6, shuffle=True, num_workers=0)
predictions = []

#model.eval()

with torch.no_grad():

    for idx,(inputs) in tqdm(enumerate(test_dataloader),total=int(len(test_dataset)/6)):

        pred = model(torch.transpose(inputs, 1, 3).cuda().float())

        predictions.append(np.argmax(pred.detach().cpu().numpy(),axis=1))



del model

predictions = np.hstack(predictions)
predictions
submission = pd.read_csv('/kaggle/input/platesv2/sample_submission.csv')

submission.label = predictions

decoder = {1:'dirty',0:'cleaned'}

submission.label = submission.label.map(decoder)
submission
submission.to_csv('submission.csv', index=False)
!rm -r plates