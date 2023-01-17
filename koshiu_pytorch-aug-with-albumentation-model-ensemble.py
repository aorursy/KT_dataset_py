import torch

import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, models

import matplotlib.pyplot as plt

import albumentations as A

from albumentations.pytorch import ToTensor

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import os



print(os.listdir("/kaggle/input/digit-recognizer"))



N_FOLDS = 5

BATCH_SIZE = 256
PATH = '/kaggle/input/digit-recognizer/'
# Checking GPU is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('Training on CPU...')

else:

    print('Training on GPU...')
# Dataset responsible for manipulating data for training as well as training tests.

class DatasetMNIST(torch.utils.data.Dataset):

    def __init__(self, data, augmentations=None):

        self.data = data

        self.augmentations = augmentations 

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        item = self.data.iloc[index]

                

        image = item[1:].values.astype(np.uint8).reshape((28, 28, 1))

        label = item[0]

        

        if self.augmentations  is not None:

            augmented = self.augmentations(image=image)   

            return augmented['image'], label

        else:

            return image, label
dataset = pd.read_csv(f'{PATH}train.csv')

dataset.head(1)
def custom_folds(dataset,n_folds=N_FOLDS):

    '''return train and valid indexies'''

    train_valid_id = []

    start = 0

    size = len(dataset)

    split = size // n_folds

    valid_size = split

    for i in range(n_folds):

        train_data = dataset.drop(dataset.index[start:split]).index.values

        valid_data = dataset.loc[start:split-1].index.values        

        train_valid_id.append((train_data,valid_data))

        start += valid_size

        split += valid_size

    return train_valid_id
train_valid = custom_folds(dataset=dataset)
transform_train = A.Compose([

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10),

    A.Normalize(mean=(0.485,), std=(0.229,)),

    ToTensor(),

])



transform_valid = A.Compose([

    A.Normalize(mean=(0.485,), std=(0.229,)),

    ToTensor(),

])
# Creating datasets for training and validation

train_data = DatasetMNIST(dataset, augmentations=transform_train)

valid_data = DatasetMNIST(dataset, augmentations=transform_valid)



# Make data loaders for N_FOLDS train/valid

train_valid_loaders = []

for i in train_valid:

    train_idx, valid_idx = i

    train_sampler = SubsetRandomSampler(train_idx)

    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, sampler=valid_sampler)

    train_valid_loaders.append((train_loader,valid_loader))
# Viewing data examples used for training (augmented data)

fig, axis = plt.subplots(2, 6, figsize=(10, 7))

# first train fold

images, labels = next(iter(train_valid_loaders[0][0]))



for i, ax in enumerate(axis.flat):

    with torch.no_grad():

        image, label = images[i], labels[i]



        ax.imshow(image.view(28, 28), cmap='binary') # add image

        ax.set(title = f"{label}") # add label
# Viewing data examples used for validation

fig, axis = plt.subplots(2, 6, figsize=(10, 7))

# first valid fold

images, labels = next(iter(train_valid_loaders[0][1]))



for i, ax in enumerate(axis.flat):

    with torch.no_grad():

        image, label = images[i], labels[i]



        ax.imshow(image.view(28, 28), cmap='binary') # add image

        ax.set(title = f"{label}") # add label
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)

        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)

        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)

        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)

        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)

        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=4)

        self.bn7 = nn.BatchNorm2d(128)

        self.lin1 = nn.Linear(128,10)

    def forward(self, xb):

        x = xb.view(-1, 1, 28, 28)

        x = self.bn1(F.relu(self.conv1(x)))

        x = self.bn2(F.relu(self.conv2(x)))

        x = self.bn3(F.relu(self.conv3(x)))

        x = F.dropout2d(x, 0.25)

        x = self.bn4(F.relu(self.conv4(x)))

        x = self.bn5(F.relu(self.conv5(x)))

        x = self.bn6(F.relu(self.conv6(x)))

        x = F.dropout2d(x, 0.25)

        x = self.bn7(F.relu(self.conv7(x)))

        x = torch.flatten(x, start_dim=1)

        x = F.dropout2d(x, 0.25)

        x = self.lin1(x)

        x = F.softmax(x, dim=1)

        return x

model = Net()

print(model)



if train_on_gpu:

    model.cuda()
# test

class DatasetSubmissionMNIST(torch.utils.data.Dataset):

    def __init__(self, file_path, augmentations=None):

        self.data = pd.read_csv(file_path)

        self.augmentations = augmentations

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        image = self.data.iloc[index].values.astype(np.uint8).reshape((28, 28, 1))



        

        if self.augmentations is not None:

            augmented = self.augmentations(image=image)   

            return augmented['image']

            

        return image
transform_test = A.Compose([

    A.Normalize(mean=(0.485,), std=(0.229,)),

    ToTensor(),

])



submissionset = DatasetSubmissionMNIST(f'{PATH}test.csv', augmentations=transform_test)

submissionloader = torch.utils.data.DataLoader(submissionset, batch_size=BATCH_SIZE, shuffle=False)
def every_predict(model,submissionloader=submissionloader):

    # my

    all_batchs = []

    with torch.no_grad():

        model.eval()

        for images in submissionloader:

            if train_on_gpu:

                images = images.cuda()

            ps = model(images)

            all_batchs.append(ps.to('cpu').detach().numpy())

    return all_batchs
five_predict = [] # all predicts

all_train_losses, all_valid_losses = [], []





FOLD = 1

for i in train_valid_loaders: # for every fold

    model = Net()

    if train_on_gpu:

        model.cuda()

    train_loader, valid_loader = i

    LEARNING_RATE = 0.01



    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    epochs = 120

    valid_loss_min = np.Inf

    train_losses, valid_losses = [], []

    history_accuracy = []

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.7,patience=2) 

    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau

    model.train()

    for e in range(1, epochs+1):

        running_loss = 0

        for images, labels in train_loader:

            if train_on_gpu:

                images, labels = images.cuda(), labels.cuda()

            # Clear the gradients, do this because gradients are accumulated.

            optimizer.zero_grad()

            # Forward pass.

            ps = model(images)

            # Calculate the loss.

            loss = criterion(ps, labels)

            # Turning loss back.

            loss.backward()

            # Take an update step and few the new weights.

            optimizer.step()

            running_loss += loss.item()

        else:

            valid_loss = 0

            accuracy = 0

            

            # Turn off gradients for validation, saves memory and computations.

            with torch.no_grad():

                model.eval() # change the network to evaluation mode

                for images, labels in valid_loader:

                    if train_on_gpu:

                        images, labels = images.cuda(), labels.cuda()

                    # Forward pass

                    ps = model(images)

                    # Capturing the class more likely.

                    _, top_class = ps.topk(1, dim=1)

                    # Verifying the prediction with the labels provided.

                    equals = top_class == labels.view(*top_class.shape)

                    valid_loss += criterion(ps, labels)

                    accuracy += torch.mean(equals.type(torch.FloatTensor))

                    

            model.train() # change the network to training mode

            train_losses.append(running_loss/len(train_loader))

            valid_losses.append(valid_loss/len(valid_loader))

            history_accuracy.append(accuracy/len(valid_loader))



            network_learned = valid_loss < valid_loss_min

            if e == 1 or e % 5 == 0 or network_learned:

                print(f"Epoch: {e}/{epochs}.. ",

                      f"Training Loss: {running_loss/len(train_loader):.4f}.. ",

                      f"Validation Loss: {valid_loss/len(valid_loader):.4f}.. ",

                      f"Valid Accuracy: {accuracy/len(valid_loader):.4f}")

            if network_learned:

                valid_loss_min = valid_loss

                torch.save(model.state_dict(), f'best_model_fold{FOLD}.pt')

                print('Detected network improvement, saving current model')

        # after every epoch        

        scheduler.step(running_loss) # put running_loss if you using ReduceLROnPlateau scheduler

        

    all_train_losses.append(train_losses)

    all_valid_losses.append(valid_losses)

    model.load_state_dict(torch.load(f'best_model_fold{FOLD}.pt')) # predict on best epoch of fold

    model.eval() # change the network to evaluation mode

    five_predict.append(every_predict(model))

    model.train() # change the network to training mode

    FOLD +=1

n_rows = int(np.ceil(N_FOLDS/3))

fig, axis = plt.subplots(n_rows, 3, figsize=(15, 10))

for i, ax in enumerate(axis.flat):

    if i<N_FOLDS:

        ax.plot(all_train_losses[i], label='Training Loss')

        ax.plot(all_valid_losses[i], label='Validation Loss')

        ax.legend(frameon=False)
flat_list = []

for sublist in five_predict:

    for item in sublist:

        for i in item:

            flat_list.append(i)

final = []

for i in range(0,28000):

    numbers = [i+a*28000 for a in range(N_FOLDS)]

    final.append(sum(flat_list[C] for C in numbers))            

subm = np.argmax((final),axis=1)

sample_subm = pd.read_csv(f'{PATH}sample_submission.csv')

sample_subm['Label'] = subm

sample_subm.to_csv('submission.csv',index=False)