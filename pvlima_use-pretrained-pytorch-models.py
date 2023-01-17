%matplotlib inline

import time

import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from PIL import Image

import torch

from torch.optim import lr_scheduler

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import torchvision

from torchvision import transforms, datasets, models
np.random.seed(0)
!ls ../input/pretrained-pytorch-models/
cache_dir = expanduser(join('~', '.torch'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)
!cp ../input/pretrained-pytorch-models/* ~/.torch/models/
!ls ~/.torch/models
!ls ../input/dog-breed-identification
INPUT_SIZE = 224

NUM_CLASSES = 16

data_dir = '../input/dog-breed-identification/'

labels = pd.read_csv(join(data_dir, 'labels.csv'))

sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))

print(len(listdir(join(data_dir, 'train'))), len(labels))

print(len(listdir(join(data_dir, 'test'))), len(sample_submission))
selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)

labels = labels[labels['breed'].isin(selected_breed_list)]

labels['target'] = 1

labels['rank'] = labels.groupby('breed').rank()['id']

labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)



train = labels_pivot.sample(frac=0.8)

valid = labels_pivot[~labels_pivot['id'].isin(train['id'])]

print(train.shape, valid.shape)
class DogsDataset(Dataset):

    def __init__(self, labels, root_dir, subset=False, transform=None):

        self.labels = labels

        self.root_dir = root_dir

        self.transform = transform

    

    def __len__(self):

        return len(self.labels)

    

    def __getitem__(self, idx):

        img_name = '{}.jpg'.format(self.labels.iloc[idx, 0])

        fullname = join(self.root_dir, img_name)

        image = Image.open(fullname)

        labels = self.labels.iloc[idx, 1:].as_matrix().astype('float')

        labels = np.argmax(labels)

        if self.transform:

            image = self.transform(image)

        return [image, labels]
normalize = transforms.Normalize(

   mean=[0.485, 0.456, 0.406],

   std=[0.229, 0.224, 0.225]

)

ds_trans = transforms.Compose([transforms.Scale(224),

                               transforms.CenterCrop(224),

                               transforms.ToTensor(),

                               normalize])

train_ds = DogsDataset(train, data_dir+'train/', transform=ds_trans)

valid_ds = DogsDataset(valid, data_dir+'train/', transform=ds_trans)



train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

valid_dl = DataLoader(valid_ds, batch_size=4, shuffle=True, num_workers=4)
def imshow(axis, inp):

    """Denormalize and show"""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    axis.imshow(inp)
img, label = next(iter(train_dl))

print(img.size(), label.size())

fig = plt.figure(1, figsize=(16, 4))

grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.05)    

for i in range(img.size()[0]):

    ax = grid[i]

    imshow(ax, img[i])
use_gpu = torch.cuda.is_available()

resnet = models.resnet50(pretrained=True)

inputs, labels = next(iter(train_dl))

if use_gpu:

    resnet = resnet.cuda()

    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())   

else:

    inputs, labels = Variable(inputs), Variable(labels)

outputs = resnet(inputs)

outputs.size()
def train_model(dataloders, model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()

    use_gpu = torch.cuda.is_available()

    best_model_wts = model.state_dict()

    best_acc = 0.0

    dataset_sizes = {'train': len(dataloders['train'].dataset), 

                     'valid': len(dataloders['valid'].dataset)}



    for epoch in range(num_epochs):

        for phase in ['train', 'valid']:

            if phase == 'train':

                scheduler.step()

                model.train(True)

            else:

                model.train(False)



            running_loss = 0.0

            running_corrects = 0



            for inputs, labels in dataloders[phase]:

                if use_gpu:

                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                else:

                    inputs, labels = Variable(inputs), Variable(labels)



                optimizer.zero_grad()



                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)



                if phase == 'train':

                    loss.backward()

                    optimizer.step()



                running_loss += loss.data[0]

                running_corrects += torch.sum(preds == labels.data)

            

            if phase == 'train':

                train_epoch_loss = running_loss / dataset_sizes[phase]

                train_epoch_acc = running_corrects / dataset_sizes[phase]

            else:

                valid_epoch_loss = running_loss / dataset_sizes[phase]

                valid_epoch_acc = running_corrects / dataset_sizes[phase]

                

            if phase == 'valid' and valid_epoch_acc > best_acc:

                best_acc = valid_epoch_acc

                best_model_wts = model.state_dict()



        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} ' 

              'valid loss: {:.4f} acc: {:.4f}'.format(

                epoch, num_epochs - 1,

                train_epoch_loss, train_epoch_acc, 

                valid_epoch_loss, valid_epoch_acc))

            

    print('Best val Acc: {:4f}'.format(best_acc))



    model.load_state_dict(best_model_wts)

    return model
resnet = models.resnet50(pretrained=True)

# freeze all model parameters

for param in resnet.parameters():

    param.requires_grad = False



# new final layer with 16 classes

num_ftrs = resnet.fc.in_features

resnet.fc = torch.nn.Linear(num_ftrs, 16)

if use_gpu:

    resnet = resnet.cuda()



criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



dloaders = {'train':train_dl, 'valid':valid_dl}
start_time = time.time()

model = train_model(dloaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs=2)

print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))
def visualize_model(dataloders, model, num_images=16):

    cnt = 0

    fig = plt.figure(1, figsize=(16, 16))

    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.05)

    for i, (inputs, labels) in enumerate(dataloders['valid']):

        if use_gpu:

            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        else:

            inputs, labels = Variable(inputs), Variable(labels)



        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)



        for j in range(inputs.size()[0]):

            ax = grid[cnt]

            imshow(ax, inputs.cpu().data[j])

            ax.text(10, 210, '{}/{}'.format(preds[j], labels.data[j]), 

                    color='k', backgroundcolor='w', alpha=0.8)

            cnt += 1

            if cnt == num_images:

                return
visualize_model(dloaders, resnet)