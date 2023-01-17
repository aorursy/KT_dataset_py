# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import copy

import os

print(os.listdir("../input"))



from tqdm import tnrange, tqdm_notebook as tqdm



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
TRAIN_PATH = "../input/train/"

train_files = list(os.listdir(TRAIN_PATH))[100:]

f = TRAIN_PATH+train_files[1]
im = mpimg.imread(f); im.shape
plt.imshow(im)

plt.show()
import torch

import torch.nn as nn

import torchvision

import torchvision.transforms as transforms
trainalldf = pd.read_csv("../input/train.csv") #, nrows=100)
trainalldf.count()
whaleids = sorted(list(trainalldf['Id'].drop_duplicates()))

print(whaleids[:5]); print(len(whaleids))
whaleids_dict = dict((k,v) for v,k in enumerate(whaleids))
BS = 32

image_input_size = 224

resnet18 = torchvision.models.resnet18(pretrained=True)

for p in resnet18.parameters():

    p.requires_grad = False # Freeze all existing layers
#model = nn.Sequential(*list(resnet18.children())[:-1], nn.Linear(512, len(whaleids)))

resnet18.fc = nn.Linear(512, len(whaleids))
resnet18.to('cuda')
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

    
trainallimages = trainalldf['Image'].values

trainallids = trainalldf['Id'].values

trainallclasses = np.array([whaleids_dict[id] for id in trainallids])
from sklearn.model_selection import ShuffleSplit
splitter = ShuffleSplit(n_splits=1, test_size=0.1)

(train_idxs, val_idxs) = next(splitter.split(trainallimages, trainallclasses))

idxs = {'train': train_idxs, 'val': val_idxs}
#train_images, train_classes = trainallimages[train_idxs], trainallclasses[train_idxs]

#val_images, val_classes = trainallimages[val_idxs], trainallclasses[val_idxs]

images_dict = {phase: trainallimages[idxs[phase]] for phase in ['train', 'val']}

classes_dict = {phase: trainallclasses[idxs[phase]] for phase in ['train', 'val']}
datasets_dict = {phase: WhaleImageDataset(TRAIN_PATH, transforms_dict[phase], images_dict[phase], classes_dict[phase]) for phase in ['train','val']}
im, c = datasets_dict['train'][1]

print(im.shape)

im = im.permute(1,2,0)

im2 = inv_normalize(im)

print(im2.shape)

plt.imshow(im2)

plt.show()
#train_dl = torch.utils.data.DataLoader(train_image_dataset, batch_size=BS, shuffle=True, num_workers=4)

#val_dl = torch.utils.data.DataLoader(val_image_dataset, batch_size=BS, shuffle=True, num_workers=4)

dataloaders_dict = {phase: torch.utils.data.DataLoader(datasets_dict[phase], batch_size=BS, shuffle=True, num_workers=1, pin_memory=True) 

                    for phase in ['train', 'val']}
#X_batch, y_batch = next(iter(dataloaders_dict['train']))
opt = torch.optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9)

crit = nn.CrossEntropyLoss()
NUM_EPOCHS = 30



val_acc_history = []



best_model_wts = copy.deepcopy(resnet18.state_dict())

best_acc = 0.0



for epoch in range(NUM_EPOCHS):

    print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))

    print('-' * 10)

    

    for phase in ['train', 'val']:

        if phase == 'train':

            resnet18.train()

        else:

            resnet18.eval()

        

        running_loss = 0.0

        running_corrects = 0

            

        for X_batch, y_batch in dataloaders_dict[phase]:

            X_batch = X_batch.to('cuda')

            y_batch = y_batch.to('cuda')

            

            opt.zero_grad()

            

            outputs = resnet18(X_batch)

            

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

            best_model_wts = copy.deepcopy(resnet18.state_dict())

        if phase == 'val':

            val_acc_history.append(epoch_acc)

        

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))     



    print('\n')

    

print('Best acc: {:.4f}'.format(best_acc))

resnet18.load_state_dict(best_model_wts)
torch.save(resnet18.state_dict(), './resnet18.model')
resnet18.load_state_dict(torch.load('./resnet18.model'))
def avprec_cutoff(inds, targets, N=5, m=1):

    rels = (inds.numpy() == targets.numpy()).astype('int')

    pks = []

    for ki in range(1,N+1):

        pk = rels[:,0:ki].sum(axis=1).reshape(-1,1)/ki

        pks.append(pk/m)



    return (np.concatenate(pks, axis=1) * rels).sum(axis=1)
npwhaleids = np.array(whaleids)

gap_num = 0.0

gap_count = 0



for x_batch, y_batch in tqdm(dataloaders_dict['val']):

    x_batch = x_batch.to('cuda')

    outputs = resnet18(x_batch)

    predinds = torch.argsort(outputs, dim=1, descending=True)[:,:5]

    

    gap_num += avprec_cutoff(predinds.to('cpu'), y_batch.view(-1,1), 5,1).sum()



    gap_count += y_batch.shape[0]



print(gap_num/gap_count)
resnet18.eval()

TEST_PATH = "../input/test/"

images_test = list(os.listdir(TEST_PATH))

dataset_test = WhaleImageDataset(TEST_PATH, transforms_dict['val'], images_test)

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BS, shuffle=False, num_workers=1, pin_memory=True)
npwhaleids = np.array(whaleids)

test_classnames = []

for test_batch in tqdm(dataloader_test):

    test_batch = test_batch.to('cuda')

    outputs = resnet18(test_batch)

    predinds = torch.argsort(outputs, dim=1, descending=True)[:,:5]

    

    whalestrs = npwhaleids[predinds.to('cpu').detach().numpy()].tolist()

    

    test_classnames.extend([" ".join(s) for s in whalestrs])

testdf = pd.DataFrame({'Image': images_test, 'Id': test_classnames})
testdf.to_csv('submission.csv', index=False)
from IPython.display import FileLink

FileLink('submission.csv')