# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm



import torch

import torch.nn as nn

from torch.optim import lr_scheduler

import torch.utils.model_zoo as model_zoo

from sklearn.metrics import roc_curve, auc

import torchvision

from sklearn.model_selection import StratifiedShuffleSplit



from PIL import Image

import matplotlib.pyplot as plt

from IPython.display import clear_output



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import random

import copy

import os



# Check if gpu support is available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



%matplotlib inline



csv_submission_ex_file = '../input/sample_submission.csv'

csv_file = '../input/train_labels.csv'

train_dir = '../input/train/'

test_dir = '../input/test/'



# Any results you write to the current directory are saved as output.
csv_pd = pd.read_csv(csv_file)   

csv_pd.describe()
class NormFinderDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, csv_file, img_dir, transform=None):

        self.csv_file = pd.read_csv(csv_file)

        self.img_dir = img_dir



        self.transform = transform



    def __len__(self):

        return len(self.csv_file)



    def __getitem__(self, idx):

        img_name = os.path.join(self.img_dir, self.csv_file.iloc[idx, 0])

        img_name = img_name + '.tif'



        sample = Image.open(img_name)



        if self.transform is not None:

            sample = self.transform(sample)



        return {'sample': sample}
normfinder_transformations = torchvision.transforms.Compose([

    torchvision.transforms.Resize(224),

    torchvision.transforms.ToTensor(),

])



normfinder_test_dataset = NormFinderDataset(csv_submission_ex_file, test_dir, normfinder_transformations)

normfinder_test_dataloader = torch.utils.data.DataLoader(normfinder_test_dataset, batch_size=1024, shuffle=False)



normfinder_train_dataset = NormFinderDataset(csv_file, train_dir, normfinder_transformations)

normfinder_train_dataloader = torch.utils.data.DataLoader(normfinder_train_dataset, batch_size=1024, shuffle=False)



pop_mean = []

pop_std0 = []

for data in tqdm(normfinder_test_dataloader, 0):

    # shape (batch_size, 3, height, width)

    numpy_image = data['sample'].numpy()



    # shape (3,)

    batch_mean = np.mean(numpy_image, axis=(0, 2, 3))

    batch_std0 = np.std(numpy_image, axis=(0, 2, 3))



    pop_mean.append(batch_mean)

    pop_std0.append(batch_std0)



for data in tqdm(normfinder_train_dataloader, 0):

    # shape (batch_size, 3, height, width)

    numpy_image = data['sample'].numpy()



    # shape (3,)

    batch_mean = np.mean(numpy_image, axis=(0, 2, 3))

    batch_std0 = np.std(numpy_image, axis=(0, 2, 3))



    pop_mean.append(batch_mean)

    pop_std0.append(batch_std0)



# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)

pop_mean = np.array(pop_mean).mean(axis=0)

pop_std0 = np.array(pop_std0).mean(axis=0)



print('mean: {}'.format(pop_mean))

print('std0: {}'.format(pop_std0))
# Let's define some transformations for the input data, crop to 64px and then resize to 224 to fit resnet input size

data_transformations = torchvision.transforms.Compose([

    torchvision.transforms.RandomHorizontalFlip(),

    torchvision.transforms.RandomVerticalFlip(),

    torchvision.transforms.Resize(224),

    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize(mean=[0.70017236, 0.5436771, 0.6961061], std=[0.22246036, 0.26757348, 0.19798167]),

])



# Let's define some transformations for the test data, crop to 32px and then resize to 224 to fit resnet input size

test_transformations = torchvision.transforms.Compose([

    torchvision.transforms.Resize(224),

    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize(mean=[0.70017236, 0.5436771, 0.6961061], std=[0.22246036, 0.26757348, 0.19798167]),

])
# let's see some negative images

fig=plt.figure(figsize=(16, 16))



samples_per_type = 5

negative_found = 0



while negative_found < samples_per_type:

    idx = random.randint(0, len(csv_pd))

    # Negatives

    if csv_pd.iloc[idx, 1] == 0:

        negative_found = negative_found + 1

        image = Image.open(train_dir + csv_pd.iloc[idx, 0] + '.tif')

        fig.add_subplot(2, 5, negative_found)

        plt.title('Negative Label')

        plt.imshow(image)

        

        image = data_transformations(image)

        back_transform = torchvision.transforms.ToPILImage()

        image = back_transform(image)

        fig.add_subplot(2, 5, negative_found + samples_per_type)

        plt.title('Transformed')

        plt.imshow(image)



plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)   

plt.show()    





# let's see some positive images

fig2=plt.figure(figsize=(16, 16))



positive_found = 0

        

while positive_found < samples_per_type:

    idx = random.randint(0, len(csv_pd))

    # Positives

    if csv_pd.iloc[idx, 1] == 1:

        positive_found = positive_found + 1

        image = Image.open(train_dir + csv_pd.iloc[idx, 0] + '.tif')

        fig2.add_subplot(2, 5, positive_found)

        plt.title('Positive Label')

        plt.imshow(image)

                

        image = data_transformations(image)

        back_transform = torchvision.transforms.ToPILImage()

        image = back_transform(image)

        fig2.add_subplot(2, 5, positive_found + samples_per_type)

        plt.title('Transformed')

        plt.imshow(image)



plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)   

plt.show()
# let's see some images after transform.

fig=plt.figure(figsize=(16, 16))



samples = 5



for i in range(samples):

    random_file = random.choice(os.listdir(test_dir))

    image = Image.open(test_dir + random_file)

    fig.add_subplot(2, 5, i + 1)

    plt.title('Original')

    plt.imshow(image)

    

    image = test_transformations(image)

    back_transform = torchvision.transforms.ToPILImage()

    image = back_transform(image)

    fig.add_subplot(2, 5, i + 1 + samples)

    plt.title('Transformed')

    plt.imshow(image)



plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)   

plt.show()
# Dataset Class for reading the data ids and labels from the CSV file.

# From the CSV file only the specified indexes are saved in this dataset(specified indexes after the train/val split)



class HPLCDDataset(torch.utils.data.dataset.Dataset):

    """HPLCDDataset dataset."""



    def __init__(self, csv_file, img_dir, idxs, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            img_dir (string): Directory with all the images.

            idxs (list): List with indexes.

            transform (callable, optional): Optional transform to be applied

                on a sample

        """

        self.csv_file = pd.read_csv(csv_file)

        self.img_dir = img_dir

        self.idxs = idxs



        self.transform = transform



    def __len__(self):

        return len(self.idxs)



    def __getitem__(self, idx):

        img_name = os.path.join(self.img_dir, self.csv_file.iloc[self.idxs[idx], 0])

        img_name = img_name + '.tif'



        sample = Image.open(img_name)

        target = self.csv_file.iloc[self.idxs[idx], 1]



        if self.transform is not None:

            sample = self.transform(sample)



        return sample, target

    

    def __getlabel__(self, idx):

        return self.csv_file.iloc[self.idxs[idx], 1]
# Use a batch size of 256 images

batch_size = 256



# Split data 10% validation 90% train with balanced data between the two datasets

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=7)



# Indexes in the CSV file for both train/val datasets

train_index, test_index = next(sss.split(csv_pd["id"], csv_pd["label"]))



train_dataset = HPLCDDataset(csv_file, train_dir, train_index, data_transformations)

test_dataset = HPLCDDataset(csv_file, train_dir, test_index, test_transformations)



# Create loders for training/validation sets

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)



dataloaders = {'train': train_loader, 'test': test_loader}
# Lets see how the data is balanced in the two Datasets

train_positive_labels = 0

train_negative_labels = 0

for index in range(train_dataset.__len__()):

    label = train_dataset.__getlabel__(index)

    if label == 0:

        train_negative_labels = train_negative_labels + 1

    else:

        train_positive_labels = train_positive_labels + 1



# Lets see how the data is balanced in the two Datasets

test_positive_labels = 0

test_negative_labels = 0

for index in range(test_dataset.__len__()):

    label = test_dataset.__getlabel__(index)

    if label == 0:

        test_negative_labels = test_negative_labels + 1

    else:

        test_positive_labels = test_positive_labels + 1



# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'Negative', 'Positive'



fig, axs = plt.subplots(1, 2)

plt.subplots_adjust(right=1.5)



def make_autopct(values):

    def my_autopct(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{p:.3f}%\n({v:d})'.format(p=pct,v=val)

    return my_autopct



axs[0].pie([train_negative_labels, train_positive_labels], labels=labels, autopct=make_autopct([train_negative_labels, train_positive_labels]),

        shadow=True, startangle=90)

axs[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

axs[0].set_title('Train Dataset\n Samples:{}'.format(train_negative_labels + train_positive_labels))



axs[1].pie([test_negative_labels, test_positive_labels], labels=labels, autopct=make_autopct([test_negative_labels, test_positive_labels]),

        shadow=True, startangle=90)

axs[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

axs[1].set_title('Test Dataset\n Samples:{}'.format(test_negative_labels + test_positive_labels))



plt.show()
def train(model, criterion, optimizer, scheduler, dataloaders, logger):

    

    dataset_sizes = {'train': len(dataloaders['train']),

                     'test': len(dataloaders['test'])}

    

    train_loss = 0.0

    train_acc = 0.0

    test_loss = 0.0

    test_acc = 0.0



    # Each epoch has a training and validation phase

    for phase in ['train', 'test']:

        if phase == 'train':

            scheduler.step()

            model.train()  # Set model to training mode

        else:

            model.eval()   # Set model to evaluate mode



        running_loss = 0.0

        running_corrects = 0

            

        test_output_results = []

        test_output_expected = []



        pbar = tqdm(enumerate(dataloaders[phase]))

        # Iterate over data.

        for i, (inputs, labels) in pbar:

            inputs = inputs.to(device)

            labels = labels.to(device)



            # zero the parameter gradients

            optimizer.zero_grad()



            # forward

            # track history if only in train

            with torch.set_grad_enabled(phase == 'train'):

                outputs = model(inputs)

                loss = criterion(outputs, labels.float().view(dataloaders[phase].batch_size, 1))

                                

                # backward + optimize only if in training phase

                if phase == 'train':

                    loss.backward()

                    optimizer.step()

                

            # statistics

            running_loss += loss.item()

            

            preds = torch.where(outputs > 0.5, torch.Tensor([1]).to(device), torch.Tensor([0]).to(device))

            running_corrects += torch.sum(preds == labels.float().view(dataloaders[phase].batch_size, 1)).item()

                

            if phase == 'test':

                test_output_results = np.concatenate([test_output_results, outputs.view(-1).cpu().numpy()])

                test_output_expected = np.concatenate([test_output_expected, labels.view(-1).cpu().numpy()])



            pbar.set_description('[{} {}/{}] Loss: {:.4f}, Acc: {:.4f}'.format(phase, i, dataset_sizes[phase],

                running_loss / (i+1), running_corrects / ((i+1) * dataloaders[phase].batch_size)))



        epoch_loss = running_loss / dataset_sizes[phase]

        epoch_acc = running_corrects / dataset_sizes[phase] / dataloaders[phase].batch_size



        if phase == 'train':

            train_loss = epoch_loss

            train_acc = epoch_acc



        if phase == 'test':

            test_loss = epoch_loss

            test_acc = epoch_acc

            logger.append([train_loss, train_acc, test_loss, test_acc])

            

        test_output = {'expected': test_output_expected, 'results': test_output_results}



    return model, optimizer, scheduler, test_output, logger
def plot_results(logger):

    plt.plot(logger)

    plt.xlabel('Epoch')

    plt.grid(True)

    plt.legend(['train loss', 'train accuracy', 'validation loss', 'validation accuracy'])

    plt.show()
def plot_roc_auc(test_output):        

    fpr = dict()

    tpr = dict()

    roc_auc = dict()   

    fpr, tpr, _ = roc_curve(test_output['expected'], test_output['results'])

    roc_auc = auc(fpr, tpr)

   

    plt.subplot(121, aspect='equal')

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.5f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    

    negative_results = []

    positive_results = []

    for idx, (res) in enumerate(test_output['expected']):

        if res == 1:

            positive_results.append(test_output['results'][idx])

        elif res == 0:

            negative_results.append(test_output['results'][idx])

        else:

            print('ERROR HIST!!!')

    

    bins = np.linspace(min(test_output['results']), max(test_output['results']), 100)

    

    plt.subplot(122)

    plt.hist(positive_results, bins, alpha=0.5, label='Positive', histtype='step')

    plt.hist(negative_results, bins, alpha=0.5, label='Negative', histtype='step')

    plt.yscale('log')

    plt.legend(loc='upper center')

    plt.grid(True)

    

    plt.subplots_adjust(bottom=0.0, right=2.2, top=1)    

    plt.show() 
model = nn.Sequential(torchvision.models.resnet18(pretrained=False, num_classes=1), nn.Sigmoid())

model.to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

criterion = nn.BCELoss()



# Decay LR by a factor of 0.1 every x epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)



# define the global logger

logger = [[0.45, 0.55, 0.45, 0.55]]



previous_epochs = 0



load_checkpoint = False

if load_checkpoint == True:

    checkpoint = torch.load("checkpoint_ep1.plt")

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    previous_epochs = checkpoint['epoch']

    dataloaders = checkpoint['dataloaders']

    logger = checkpoint['logger']
epochs = 8



for i in range(epochs):

    print('Epoch: {}/{}'.format(i + previous_epochs + 1, epochs + previous_epochs))

    model, optimizer, exp_lr_scheduler, test_output, logger = train(model, criterion, optimizer, exp_lr_scheduler, dataloaders, logger)



    #SAVE

    #torch.save({

    #        'epoch': i + 1 + previous_epochs,

    #        'model_state_dict': model.state_dict(),

    #        'optimizer_state_dict': optimizer.state_dict(),

    #        'logger': logger,

    #        'dataloaders': dataloaders

    #        }, "checkpoint_ep{}.plt".format(i + 1 + previous_epochs))

    

    clear_output()
plot_roc_auc(test_output) 

plot_results(logger)