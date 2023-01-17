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
import sys

import time

import math

import gc

import cv2



import torch

import torch.nn as nn

from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader

import torchvision

import torchvision.transforms as T 

import torchvision.models as models

import matplotlib.pyplot as plt



import sklearn.metrics



def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



root_train = '/kaggle/input/bengaliaicv19feather'

root_test = '/kaggle/input/bengaliai-cv19'



#pretrained = '/kaggle/input/trained-model/new_resnet2.pth'

# resnet.pth -->  T.Compose([T.ToPILImage(), T.CenterCrop(size), T.ToTensor()])

# new_resnet.pth --> T.Compose([T.ToPILImage(), T.CenterCrop(150), T.Resize((size,size)), T.RandomAffine(45), T.ToTensor()])



path = 'new_resnet34.pth'

# keeping training by adding noise in dataset
start = time.time()

img_arrs = np.concatenate((pd.read_feather(os.path.join(root_train,'train_image_data_0.feather')).drop(columns = ['image_id']).values, 

                           pd.read_feather(os.path.join(root_train,'train_image_data_1.feather')).drop(columns = ['image_id']).values, 

                           pd.read_feather(os.path.join(root_train,'train_image_data_2.feather')).drop(columns = ['image_id']).values, 

                           pd.read_feather(os.path.join(root_train,'train_image_data_3.feather')).drop(columns = ['image_id']).values), 

                          axis=0)

print(timeSince(start))

print(img_arrs.shape)
#idx = 38

#threshold = 0.5

#size = 128



#img_arr = img_arrs[idx]/255.

#fig = plt.figure(figsize=(20, 6))



#ax = fig.add_subplot(2, 3, 1, xticks=[], yticks=[])

#original_img = img_arr.reshape(137,236)

#ax.imshow(original_img, cmap='gray')



#ax = fig.add_subplot(2, 3, 2, xticks=[], yticks=[])

#cropped_img = original_img[:, 30:190]

#ax.imshow(cropped_img, cmap='gray')



#ax = fig.add_subplot(2, 3, 3, xticks=[], yticks=[])

#resized = cv2.resize(cropped_img,(size,size))

#ax.imshow(resized, cmap='gray')





#ax = fig.add_subplot(2, 3, 4, xticks=[], yticks=[])

#rotate = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)

#ax.imshow(rotate, cmap='gray')



#ax = fig.add_subplot(2, 3, 5, xticks=[], yticks=[])

#rotate = cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE)

#ax.imshow(rotate, cmap='gray')



#ax = fig.add_subplot(2, 3, 6, xticks=[], yticks=[])

#rotate = cv2.rotate(resized, cv2.ROTATE_180)

#ax.imshow(rotate, cmap='gray')



#plt.show()
#idx = 653

#size = 128

#threshold = 0.5



#img_arr = 255 - img_arrs[idx] # flip black and white, so the default padding value (0) could match

#original_img = img_arr.reshape(137,236,1)



#fig = plt.figure(figsize=(20, 6))

#transforms = T.Compose([T.ToPILImage(), T.ToTensor()])

#img_tensor = transforms(original_img)

#ax = fig.add_subplot(2, 3, 1, xticks=[], yticks=[])

#ax.imshow(img_tensor[0], cmap='gray')



#transforms = T.Compose([T.ToPILImage(),T.RandomAffine(90),T.CenterCrop(150), T.Resize((size,size)),T.ToTensor()])

#img_tensor = transforms(original_img)

#ax = fig.add_subplot(2, 3, 2, xticks=[], yticks=[])

#ax.imshow(img_tensor[0], cmap='gray')



#img_tensor = transforms(original_img)

#ax = fig.add_subplot(2, 3, 3, xticks=[], yticks=[])

#ax.imshow(img_tensor[0], cmap='gray')



#img_tensor = transforms(original_img)

#ax = fig.add_subplot(2, 3, 4, xticks=[], yticks=[])

#ax.imshow(img_tensor[0], cmap='gray')



#img_tensor = transforms(original_img)

#ax = fig.add_subplot(2, 3, 5, xticks=[], yticks=[])

#ax.imshow(img_tensor[0], cmap='gray')



#img_tensor = transforms(original_img)

#ax = fig.add_subplot(2, 3, 6, xticks=[], yticks=[])

#ax.imshow(img_tensor[0], cmap='gray')



##new_tensor = torch.where(img_tensor <= threshold, torch.zeros_like(img_tensor), img_tensor)

##plt.imshow(new_tensor[0], cmap='gray')

#plt.show()
class graphemeDataset(Dataset):

    def __init__(self, img_arrs, target_file = None):

        self.img_arrs = img_arrs

        self.target_file = target_file

        

        if target_file is None:

            self.transforms = T.Compose([T.ToPILImage(), T.CenterCrop(150), T.Resize((128,128)),T.ToTensor()])

        else:

            self.transforms = T.Compose([T.ToPILImage(),T.RandomAffine(90),T.CenterCrop(150), T.Resize((128,128)),T.ToTensor()])

            # add targets for training

            target_df = pd.read_csv(target_file)

            self.grapheme = target_df['grapheme_root'].values

            self.vowel = target_df['vowel_diacritic'].values

            self.consonant = target_df['consonant_diacritic'].values

            del target_df

            gc.collect()

               

    def __getitem__(self, idx):

        img_arr = 255 - self.img_arrs[idx] # flip black and white, so the default padding value (0) could match

        new_tensor = self.transforms(img_arr.reshape(137, 236, 1))

        

        if self.target_file is None:

            return new_tensor

        else:

            grapheme_tensor = torch.tensor(self.grapheme[idx], dtype=torch.long)

            vowel_tensor = torch.tensor(self.vowel[idx], dtype=torch.long)

            consonant_tensor = torch.tensor(self.consonant[idx], dtype=torch.long)

            return new_tensor, grapheme_tensor, vowel_tensor, consonant_tensor

    

    def __len__(self):

        return len(self.img_arrs)
dataset = graphemeDataset(img_arrs, target_file = '/kaggle/input/bengaliai-cv19/train.csv')

print(dataset.__len__())



batch_size = 128

data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

dataiter = iter(data_loader)

print(round(dataset.__len__()/batch_size))
img_tensor, grapheme_tensor, vowel_tensor, consonant_tensor = next(dataiter)



print(img_tensor.size())

print(grapheme_tensor.size())

print(vowel_tensor.size())

print(consonant_tensor.size())



fig = plt.figure(figsize=(25, 8))

plot_size = 32

for idx in np.arange(plot_size):

    ax = fig.add_subplot(4, plot_size/4, idx+1, xticks=[], yticks=[])

    ax.imshow(img_tensor[idx][0], cmap='gray')
class simpleCNN(nn.Module):

    def __init__(self, hidden_dim = 32):

        super(simpleCNN, self).__init__()

        self.hidden_dim = hidden_dim



        self.features = nn.Sequential(

            nn.Conv2d(1, hidden_dim, kernel_size = 3),

            nn.ReLU(inplace = True),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3),

            nn.ReLU(inplace = True),

            nn.BatchNorm2d(hidden_dim),

            nn.MaxPool2d(2),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 5),

            nn.ReLU(inplace = True),

            nn.Dropout(p = 0.3),

            

            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size = 3),

            nn.ReLU(inplace = True),

            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size = 3),

            nn.ReLU(inplace = True),

            nn.BatchNorm2d(hidden_dim*2),

            nn.MaxPool2d(2),

            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size = 5),

            nn.BatchNorm2d(hidden_dim*2),

            nn.Dropout(p = 0.3),

            

            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size = 3),

            nn.ReLU(inplace = True),

            nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size = 3),

            nn.ReLU(inplace = True),

            nn.BatchNorm2d(hidden_dim*4),

            nn.MaxPool2d(2),

            nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size = 5),

            nn.BatchNorm2d(hidden_dim*4),

            nn.Dropout(p = 0.3),

            

            nn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size = 3),

            nn.ReLU(inplace = True),

            nn.Conv2d(hidden_dim*8, hidden_dim*8, kernel_size = 3),

            nn.ReLU(inplace = True),

            nn.BatchNorm2d(hidden_dim*8),

            nn.Dropout(p = 0.3),

            

            nn.Flatten()

        )

        

        self.grapheme_classifier = nn.Sequential(

            nn.Linear(self.hidden_dim*8, 168),

            nn.LogSoftmax(dim=1)

        )



        self.vowel_classifier = nn.Sequential(

            nn.Linear(self.hidden_dim*8, self.hidden_dim*4),

            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_dim*4, 11),

            nn.LogSoftmax(dim=1)

        )

        

        self.consonant_classifier = nn.Sequential(

            nn.Linear(self.hidden_dim*8, self.hidden_dim*4),

            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_dim*4, 7),

            nn.LogSoftmax(dim=1)

        )

        

    def forward(self, x):

        x = self.features(x)

        c1 = self.grapheme_classifier(x)

        c2 = self.vowel_classifier(x)

        c3 = self.consonant_classifier(x)

        return c1, c2, c3

        #return x
class modified_resnet50(nn.Module):

    def __init__(self):

        super(modified_resnet50, self).__init__()

        

        resnet50 = models.resnet50()

        resnet50.conv1 =  nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)

        layers = list(resnet50.children())[:-1] + [nn.Flatten()]

        self.features= nn.Sequential(*layers)

        

        self.grapheme_classifier = nn.Sequential(

            nn.Linear(2048, 168),

            nn.LogSoftmax(dim=1)

        )



        self.vowel_classifier = nn.Sequential(

            nn.Linear(2048, 11),

            nn.LogSoftmax(dim=1)

        )

        

        self.consonant_classifier = nn.Sequential(

            nn.Linear(2048, 7),

            nn.LogSoftmax(dim=1)

        )

        

    def forward(self, x):

        x = self.features(x)

        c1 = self.grapheme_classifier(x)

        c2 = self.vowel_classifier(x)

        c3 = self.consonant_classifier(x)

        return c1, c2, c3
class modified_resnet34(nn.Module):

    def __init__(self):

        super(modified_resnet34, self).__init__()

        

        resnet34 = models.resnet34()

        resnet34.conv1 =  nn.Conv2d(1, 64, kernel_size = 5, stride = 1, padding = 2, bias = False)

        layers = list(resnet34.children())[:-1] + [nn.Flatten()]

        self.features= nn.Sequential(*layers)

        

        self.grapheme_classifier = nn.Sequential(

            nn.Linear(512, 168),

            nn.LogSoftmax(dim=1)

        )



        self.vowel_classifier = nn.Sequential(

            nn.Linear(512, 11),

            nn.LogSoftmax(dim=1)

        )

        

        self.consonant_classifier = nn.Sequential(

            nn.Linear(512, 7),

            nn.LogSoftmax(dim=1)

        )

        

    def forward(self, x):

        x = self.features(x)

        c1 = self.grapheme_classifier(x)

        c2 = self.vowel_classifier(x)

        c3 = self.consonant_classifier(x)

        return c1, c2, c3
#model = modified_resnet34()

#c1, c2, c3 = model(img_tensor)

#print(c1.size())

#print(c2.size())

#print(c3.size())
def model_training(model, dataset, path, batch_size, epoches, print_every):

    start = time.time()

    

    criterion = nn.NLLLoss()

    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

    

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)

    

    losses = []

    loss_hold = 999

    

    for e in range(epoches):

        for counter, (img_tensor, grapheme_tensor, vowel_tensor, consonant_tensor) in enumerate(data_loader):

            img_tensor = img_tensor.to(device)

            grapheme_tensor = grapheme_tensor.to(device)

            vowel_tensor = vowel_tensor.to(device)

            consonant_tensor = consonant_tensor.to(device)

            c1, c2, c3 = model(img_tensor)

            

            l1 = criterion(c1, grapheme_tensor)

            l2 = criterion(c2, vowel_tensor)

            l3 = criterion(c3, consonant_tensor)

            loss = 0.5* l1 + 0.25*l2 + 0.25*l3

            

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            

            if counter % print_every == 0:

                print('Epoch {}/{}({}) {}...loss {:6.4f}...grapheme {:6.4f}...vowel {:6.4f}...consonant {:6.4f}'.format(

                    e+1, epoches, counter, timeSince(start), loss.item(), l1.item(),l2.item(),l3.item()))

                losses.append((loss.item(), l1.item(),l2.item(),l3.item()))

                

                if loss.item() < loss_hold:

                    torch.save(model.state_dict(), path)

                    loss_hold = loss.item()

                    print('Check point saved...')

         

        exp_lr_scheduler.step()

    return losses
model = modified_resnet34().to(device)

#model.load_state_dict(torch.load(pretrained))

#model.to(device)
losses = model_training(model, dataset, path, batch_size = 128, epoches = 21, print_every = 500)
losses = np.array(losses)

plt.figure(figsize=(10,5))

plt.plot(losses.T[0], label='weighted loss')

plt.plot(losses.T[1], label='grapheme')

plt.plot(losses.T[2], label='vowel')

plt.plot(losses.T[3], label='consonant')

plt.xlabel("iterations")

plt.ylabel("Loss")

plt.legend()

plt.show()
# sample part of the training data to check training accuracy

#start = time.time()

#loader = DataLoader(dataset, batch_size = 1024, shuffle = True)

#dataiter = iter(loader)

#img_tensor, grapheme_tensor, vowel_tensor, consonant_tensor = dataiter.next()



#model = modified_resnet()

#model.load_state_dict(torch.load(path))

#model.to(device)

#with torch.no_grad():

#    img_tensor = img_tensor.to(device)

#    c1, c2, c3 = model(img_tensor)

    

#grapheme_pred = c1.argmax(1).cpu().tolist()

#vowel_pred = c2.argmax(1).cpu().tolist()

#consonant_pred = c3.argmax(1).cpu().tolist()



#grapheme_true =grapheme_tensor.tolist()

#vowel_true = vowel_tensor.tolist()

#consonant_true = consonant_tensor.tolist()

        

#scores = [sklearn.metrics.recall_score(grapheme_true, grapheme_pred, average='macro'),

#          sklearn.metrics.recall_score(vowel_true, vowel_pred, average='macro'),

#          sklearn.metrics.recall_score(consonant_true, consonant_pred, average='macro')]

#final_score = np.average(scores, weights=[2,1,1])

#print('train acc: {:6.2f} %'.format(100*final_score))

#print(timeSince(start))
# cleanup to release memory

del img_arrs

gc.collect()