#github link to this repo is https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/Pytorch%20transfer%20learning%20tutorial%20%5B93%25acc%5D.ipynb

!pip install torchsummary

!pip install efficientnet_pytorch

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn import metrics

from torchvision import transforms

from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset

import pandas as pd

import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler

import cv2

import os

import torchvision

import shutil

from torch.autograd import Variable

import torch

from torch import nn

import torch.nn.functional as F

import torchvision.models as models

from torchsummary import summary

from efficientnet_pytorch import EfficientNet

import torch.optim as optim

from tqdm.autonotebook import tqdm

# Any results you write to the current directory are saved as output.
#parameters

batch_size = 8

im_size = 150

def normalization_parameter(dataloader):

    mean = 0.

    std = 0.

    nb_samples = len(dataloader.dataset)

    for data,_ in tqdm(dataloader):

        batch_samples = data.size(0)

        data = data.view(batch_samples, data.size(1), -1)

        mean += data.mean(2).sum(0)

        std += data.std(2).sum(0)

    mean /= nb_samples

    std /= nb_samples

    return mean.numpy(),std.numpy()

im_size = 150

train_transforms = transforms.Compose([

                                        transforms.Resize((im_size,im_size)),

                                        transforms.ToTensor()])

train_data = torchvision.datasets.ImageFolder(root = '../input/seg_train/seg_train', transform = train_transforms)

train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)

mean,std = normalization_parameter(train_loader)
#image transformations for train and test data



train_transforms = transforms.Compose([

                                        transforms.Resize((im_size,im_size)),

                                        transforms.RandomResizedCrop(size=315, scale=(0.95, 1.0)),

                                        transforms.RandomRotation(degrees=10),

                                        transforms.RandomHorizontalFlip(),

                                        transforms.CenterCrop(size=299),  # Image net standards

                                        transforms.ToTensor(),

                                        transforms.Normalize(mean,std)])

test_transforms = transforms.Compose([

                                        transforms.Resize((im_size,im_size)),

                                        transforms.ToTensor(),

                                        transforms.Normalize(mean,std)])



#inverse normalization for image plot



inv_normalize =  transforms.Normalize(

    mean=-1*np.divide(mean,std),

    std=1/std

)
def data_loader(train_data,test_data = None , valid_size = None , batch_size = 32):

    train_loader =  DataLoader(train_data, batch_size = batch_size , shuffle = True)

    if(test_data == None and valid_size == None):

        print('c')

        dataloaders = {'train':train_loader}

        return dataloaders

    if(test_data == None and valid_size!=None):

        data_len = len(train_data)

        indices = list(range(data_len))

        np.random.shuffle(indices)

        split1 = int(np.floor(valid_size * data_len))

        valid_idx , test_idx = indices[:split1], indices[split1:]

        valid_sampler = SubsetRandomSampler(valid_idx)

        valid_loader = DataLoader(train_data, batch_size= batch_size, sampler=valid_sampler)

        dataloaders = {'train':train_loader,'val':valid_loader}

        return dataloaders

    if(test_data != None and valid_size!=None):

        data_len = len(test_data)

        indices = list(range(data_len))

        np.random.shuffle(indices)

        split1 = int(np.floor(valid_size * data_len))

        valid_idx , test_idx = indices[:split1], indices[split1:]

        valid_sampler = SubsetRandomSampler(valid_idx)

        test_sampler = SubsetRandomSampler(test_idx)

        valid_loader = DataLoader(test_data, batch_size= batch_size, sampler=valid_sampler)

        test_loader = DataLoader(test_data, batch_size= batch_size, sampler=test_sampler)

        dataloaders = {'train':train_loader,'val':valid_loader,'test':test_loader}

        return dataloaders
#data loader

train_data = torchvision.datasets.ImageFolder(root = '../input/seg_train/seg_train', transform = train_transforms)

test_data = torchvision.datasets.ImageFolder(root = '../input/seg_test/seg_test', transform = test_transforms)

dataloaders = data_loader(train_data,test_data , valid_size = 0.2 , batch_size = batch_size)

#label of classes

classes = train_data.classes

#encoder and decoder to convert classes into integer

decoder = {}

for i in range(len(classes)):

    decoder[classes[i]] = i

encoder = {}

for i in range(len(classes)):

    encoder[i] = classes[i]
import matplotlib.pyplot as plt

import random

#plotting rondom images from dataset

def class_plot(data , encoder ,inv_normalize = None,n_figures = 12):

    n_row = int(n_figures/4)

    fig,axes = plt.subplots(figsize=(14, 10), nrows = n_row, ncols=4)

    for ax in axes.flatten():

        a = random.randint(0,len(data))

        (image,label) = data[a]

        label = int(label)

        l = encoder[label]

        if(inv_normalize!=None):

            image = inv_normalize(image)

        

        image = image.numpy().transpose(1,2,0)

        im = ax.imshow(image)

        ax.set_title(l)

        ax.axis('off')

    plt.show()

class_plot(train_data,encoder,inv_normalize)
#using efficientnet model based transfer learning

class Classifier(nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()

        self.resnet =  EfficientNet.from_pretrained('efficientnet-b3')

        self.l1 = nn.Linear(1000 , 256)

        self.dropout = nn.Dropout(0.75)

        self.l2 = nn.Linear(256,6)

        self.relu = nn.ReLU()



    def forward(self, input):

        x = self.resnet(input)

        x = x.view(x.size(0),-1)

        x = self.dropout(self.relu(self.l1(x)))

        x = self.l2(x)

        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = Classifier().to(device)
# Loss -> Negative log likelihood loss if output layer logsoftmax else for linear layer we use crossentropy loss.

criterion = nn.CrossEntropyLoss()

#lr scheduler ->

#learning rate half after 3 epochs

# cyclical learning rate ->

#Original learning rate restored after 10 epochs
import numpy as np

import torch

class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):

        """

        Args:

            patience (int): How long to wait after last time validation loss improved.

                            Default: 7

            verbose (bool): If True, prints a message for each validation loss improvement. 

                            Default: False

        """

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf



    def __call__(self, val_loss, model):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0



    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), 'checkpoint.pt')

        self.val_loss_min = val_loss
!git clone https://github.com/davidtvs/pytorch-lr-finder
shutil.copy('./pytorch-lr-finder/lr_finder.py','./lr_finder.py')
'''from lr_finder import LRFinder

optimizer_ft = optim.Adam(classifier.parameters(), lr=0.0000001)

lr_finder = LRFinder(classifier, optimizer_ft, criterion, device=device)

lr_finder.range_test(train_loader, end_lr=1, num_iter=500)

lr_finder.reset()

lr_finder.plot()'''
def train(model,dataloaders,criterion,num_epochs=10,lr=0.00001,batch_size=8,patience = None):

    since = time.time()

    model.to(device)

    best_acc = 0.0

    i = 0

    phase1 = dataloaders.keys()

    losses = list()

    acc = list()

    if(patience!=None):

        earlystop = EarlyStopping(patience = patience,verbose = True)

    for epoch in range(num_epochs):

        print('Epoch:',epoch)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        lr = lr*0.8

        if(epoch%10==0):

            lr = 0.0001



        for phase in phase1:

            if phase == ' train':

                model.train()

            else:

                model.eval()

            running_loss = 0.0

            running_corrects = 0

            total = 0

            j = 0

            for  batch_idx, (data, target) in enumerate(dataloaders[phase]):

                data, target = Variable(data), Variable(target)

                data = data.type(torch.cuda.FloatTensor)

                target = target.type(torch.cuda.LongTensor)

                optimizer.zero_grad()

                output = model(data)

                loss = criterion(output, target)

                _, preds = torch.max(output, 1)

                running_corrects = running_corrects + torch.sum(preds == target.data)

                running_loss += loss.item() * data.size(0)

                j = j+1

                if(phase =='train'):

                    loss.backward()

                    optimizer.step()



                if batch_idx % 300 == 0:

                    print('{} Epoch: {}  [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}'.format(phase,epoch, batch_idx * len(data), len(dataloaders[phase].dataset),100. * batch_idx / len(dataloaders[phase])

                                                                                                 , running_loss/(j*batch_size),running_corrects.double()/(j*batch_size)))

            epoch_acc = running_corrects.double()/(len(dataloaders[phase])*batch_size)

            epoch_loss = running_loss/(len(dataloaders[phase])*batch_size)

            if(phase == 'val'):

                earlystop(epoch_loss,model)



            if(phase == 'train'):

                losses.append(epoch_loss)

                acc.append(epoch_acc)

            print(earlystop.early_stop)

        if(earlystop.early_stop):

            print("Early stopping")

            model.load_state_dict(torch.load('./checkpoint.pt'))

            break

        print('{} Accuracy: '.format(phase),epoch_acc.item())

    return losses,acc
def test(dataloader):

    running_corrects = 0

    running_loss=0

    pred = []

    true = []

    pred_wrong = []

    true_wrong = []

    image = []

    sm = nn.Softmax(dim = 1)

    for batch_idx, (data, target) in enumerate(dataloader):

        data, target = Variable(data), Variable(target)

        data = data.type(torch.cuda.FloatTensor)

        target = target.type(torch.cuda.LongTensor)

        classifier.eval()

        output = classifier(data)

        loss = criterion(output, target)

        output = sm(output)

        _, preds = torch.max(output, 1)

        running_corrects = running_corrects + torch.sum(preds == target.data)

        running_loss += loss.item() * data.size(0)

        preds = preds.cpu().numpy()

        target = target.cpu().numpy()

        preds = np.reshape(preds,(len(preds),1))

        target = np.reshape(target,(len(preds),1))

        data = data.cpu().numpy()

        

        for i in range(len(preds)):

            pred.append(preds[i])

            true.append(target[i])

            if(preds[i]!=target[i]):

                pred_wrong.append(preds[i])

                true_wrong.append(target[i])

                image.append(data[i])

      

    epoch_acc = running_corrects.double()/(len(dataloader)*batch_size)

    epoch_loss = running_loss/(len(dataloader)*batch_size)

    print(epoch_acc,epoch_loss)

    return true,pred,image,true_wrong,pred_wrong
def error_plot(loss):

    plt.figure(figsize=(10,5))

    plt.plot(loss)

    plt.title("Training loss plot")

    plt.xlabel("epochs")

    plt.ylabel("Loss")

    plt.show()

def acc_plot(acc):

    plt.figure(figsize=(10,5))

    plt.plot(acc)

    plt.title("Training accuracy plot")

    plt.xlabel("epochs")

    plt.ylabel("accuracy")

    plt.show()

# To plot the wrong predictions given by model

def wrong_plot(n_figures,true,ima,pred,encoder,inv_normalize):

    print('Classes in order Actual and Predicted')

    n_row = int(n_figures/3)

    fig,axes = plt.subplots(figsize=(14, 10), nrows = n_row, ncols=3)

    for ax in axes.flatten():

        a = random.randint(0,len(true)-1)

    

        image,correct,wrong = ima[a],true[a],pred[a]

        image = torch.from_numpy(image)

        correct = int(correct)

        c = encoder[correct]

        wrong = int(wrong)

        w = encoder[wrong]

        f = 'A:'+c + ',' +'P:'+w

        if inv_normalize !=None:

            image = inv_normalize(image)

        image = image.numpy().transpose(1,2,0)

        im = ax.imshow(image)

        ax.set_title(f)

        ax.axis('off')

    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = metrics.confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

def performance_matrix(true,pred):

    precision = metrics.precision_score(true,pred,average='macro')

    recall = metrics.recall_score(true,pred,average='macro')

    accuracy = metrics.accuracy_score(true,pred)

    f1_score = metrics.f1_score(true,pred,average='macro')

    print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision*100,recall*100,accuracy*100,f1_score*100))
def train_model(model,dataloaders,criterion,num_epochs=10,lr=0.0001,batch_size=8,patience = None,classes = None):

    dataloader_train = {}

    losses = list()

    accuracy = list()

    key = dataloaders.keys()

    for phase in key:

        if(phase == 'test'):

            perform_test = True

        else:

            dataloader_train.update([(phase,dataloaders[phase])])

    losses,accuracy = train(model,dataloader_train,criterion,num_epochs,lr,batch_size,patience)

    error_plot(losses)

    acc_plot(accuracy)

    if(perform_test == True):

        true,pred,image,true_wrong,pred_wrong = test(dataloaders['test'])

        wrong_plot(12,true_wrong,image,pred_wrong,encoder,inv_normalize)

        performance_matrix(true,pred)

        if(classes !=None):

            plot_confusion_matrix(true, pred, classes= classes,title='Confusion matrix, without normalization')
train_model(classifier,dataloaders,criterion,50, patience = 5 , batch_size = batch_size , classes = classes)