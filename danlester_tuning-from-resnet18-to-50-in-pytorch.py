# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import copy

import os

DATA_PATH = "../input/humpback-whale-identification"

print(os.listdir(DATA_PATH))



from tqdm import tnrange, tqdm_notebook as tqdm



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
TRAIN_PATH = DATA_PATH+"/train/"

train_files = list(os.listdir(TRAIN_PATH))[100:]

f = TRAIN_PATH+train_files[1]
im = mpimg.imread(f); im.shape
plt.imshow(im)

plt.show()
import torch

import torch.nn as nn

import torchvision

import torchvision.transforms as transforms

from sklearn.model_selection import ShuffleSplit
trainalldf = pd.read_csv(DATA_PATH+"/train.csv", nrows=64)
trainalldf.count()
whaleids = sorted(list(trainalldf['Id'].drop_duplicates()))

print(whaleids[:5]); print(len(whaleids))
whaleids_dict = dict((k,v) for v,k in enumerate(whaleids))
BS = 32

image_input_size = 224
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inv_normalize = transforms.Normalize(

    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],

    std=[1/0.229, 1/0.224, 1/0.255]

)



transforms_dict = {

    'train': transforms.Compose([transforms.RandomResizedCrop(image_input_size),

                                 transforms.RandomHorizontalFlip(),

                                 transforms.ToTensor(),

                                 norm]),

    'val': transforms.Compose([transforms.Resize(image_input_size),

                                 transforms.CenterCrop(image_input_size),

                                 transforms.ToTensor(),

                                 norm])

}

class WhaleImageDataset(torchvision.datasets.folder.ImageFolder):

    def __init__(self, ROOT_PATH, tfm, images, targets=None):

        self.ROOT_PATH = ROOT_PATH

        self.images = images

        self.targets = targets

        self.trans = tfm

        self.loader = torchvision.datasets.folder.default_loader

    

    def __getitem__(self, index):

        f = self.ROOT_PATH + self.images[index]

        im = self.loader(f)

        if self.targets is None: # Test mode has no targets

            return self.trans(im)

        return self.trans(im), self.targets[index]

    

    def __len__(self):

        return len(self.images)

    
def split_into_train_val(trainalldf, whaleids_dict, test_size=None, train_size=None, batch_size=BS):

    trainallimages = trainalldf['Image'].values

    trainallids = trainalldf['Id'].values

    trainallclasses = np.array([whaleids_dict[id] for id in trainallids])

    

    splitter = ShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size)

    (train_idxs, val_idxs) = next(splitter.split(trainallimages, trainallclasses))

    idxs = {'train': train_idxs, 'val': val_idxs}

    

    images_dict = {phase: trainallimages[idxs[phase]] for phase in ['train', 'val']}

    classes_dict = {phase: trainallclasses[idxs[phase]] for phase in ['train', 'val']}

    

    datasets_dict = {phase: WhaleImageDataset(TRAIN_PATH, transforms_dict[phase], images_dict[phase], classes_dict[phase]) for phase in ['train','val']}

    

    dataloaders_dict = {phase: torch.utils.data.DataLoader(datasets_dict[phase], batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True) 

                    for phase in ['train', 'val']}

    

    return dataloaders_dict
# Try to overfit just one batch

dataloaders_dict = split_into_train_val(trainalldf, whaleids_dict, test_size=32, train_size=32)
#im, c = datasets_dict['train'][1]

#print(im.shape)

#im = im.permute(1,2,0)

#im2 = inv_normalize(im)

#print(im2.shape)

#plt.imshow(im2)

#plt.show()
def avprec_cutoff(inds, targets, N=5, m=1):

    rels = (inds.numpy() == targets.numpy()).astype('int')

    pks = []

    for ki in range(1,N+1):

        pk = rels[:,0:ki].sum(axis=1).reshape(-1,1)/ki

        pks.append(pk/m)



    return (np.concatenate(pks, axis=1) * rels).sum(axis=1)
def train_model(model, opt, crit, NUM_EPOCHS, dataloaders_dict, choose_best_acc=False, freeze_bn=False):

    val_acc_history = []



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(NUM_EPOCHS):

        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))

        print('-' * 10)



        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()

                if freeze_bn:

                    def set_bn_eval(m):

                        classname = m.__class__.__name__

                        if classname.find('BatchNorm') != -1:

                          m.eval()



                    model.apply(set_bn_eval)

                

            else:

                model.eval()



            running_loss = 0.0

            running_corrects = 0



            for X_batch, y_batch in dataloaders_dict[phase]:

                X_batch = X_batch.to('cuda')

                y_batch = y_batch.to('cuda')



                opt.zero_grad()



                outputs = model(X_batch)



                loss = crit(outputs, y_batch)



                _, preds = torch.max(outputs, 1)



                if phase == 'train':

                    loss.backward()

                    opt.step()



                running_loss += loss.item() * X_batch.size(0)

                running_corrects += torch.sum(preds == y_batch.data)



            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)

            epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)



            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':

                val_acc_history.append(epoch_acc)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))     



        print('\n')



    print('Best acc: {:.4f}'.format(best_acc))

    if choose_best_acc:

        print('Loading best weights')

        model.load_state_dict(best_model_wts)
resnet50 = torchvision.models.resnet50(pretrained=True)

for p in resnet50.parameters():

    p.requires_grad = False # Freeze all existing layers
resnet50.fc = nn.Linear(2048, len(whaleids))
resnet50.to('cuda')
opt = torch.optim.Adam(resnet50.fc.parameters(), lr=3e-4)

crit = nn.CrossEntropyLoss()
train_model(resnet50, opt, crit, 50, dataloaders_dict, freeze_bn=True)
for p in resnet50.parameters():

    p.requires_grad = True # Unfreeze all layers
opt = torch.optim.Adam(resnet50.fc.parameters(), lr=3e-5)

crit = nn.CrossEntropyLoss()

train_model(resnet50, opt, crit, 50, dataloaders_dict, freeze_bn=True)
x_batch, y_batch = next(iter(dataloaders_dict['train']))
y_batch
preds = resnet50(x_batch.to('cuda')); preds.max(dim=1)
freqs = trainalldf['Id'].value_counts() ; freqs[:5]
wts = [1/freqs[w] for w in whaleids] ; wts[:5]
opt = torch.optim.Adam(resnet50.fc.parameters(), lr=3e-4)

crit = nn.CrossEntropyLoss(weight=torch.Tensor(wts).to('cuda'))

train_model(resnet50, opt, crit, 50, dataloaders_dict, freeze_bn=True)