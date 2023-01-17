import os

import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import optim

from torchvision import datasets, models

import torchvision.transforms as transforms

from datetime import datetime

from torch.utils import data

import random
#transform the images to the default input shape 224x224

transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

data_set = datasets.ImageFolder(root = '../input/facemask-detection-dataset-20000-images', transform = transform)

n = len(data_set)

n_test = int( n * .2 )

n_train = n - n_test

# train, test split

train_set, test_set = data.random_split(data_set, (n_train, n_test))
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = True)
#Select 10 random images of each class.

path = '../input/facemask-detection-dataset-20000-images/'

num_files_per_folder = [len(files) for root, dirs, files in os.walk('../input/facemask-detection-dataset-20000-images') if len(files) > 0]

folders = [dirs for root, dirs, files in os.walk('../input/facemask-detection-dataset-20000-images') if len(dirs) > 0][0]

filenames = [os.listdir('../input/facemask-detection-dataset-20000-images/' + folder) for folder in folders]

files = [np.random.choice(files, 10, replace = False) for files, num_files in zip(filenames, num_files_per_folder)]
fig, ax = plt.subplots(2, 4, figsize = (15, 8))

for row in range(2):

    category = folders[row]

    ax_row = ax[row]

    for column in range(4):

        img = plt.imread(path + category + '/' + files[row][column])

        ax_column = ax_row[column]

        ax_column.imshow(img, cmap='gray')

        if column == 0:

            ax_column.set_ylabel(category, size = 'large')

        ax_column.set_xticklabels([])

        ax_column.set_yticklabels([])
#downloading the pretrained model

model = models.vgg16(pretrained = True)
print(model)
from IPython.display import YouTubeVideo



YouTubeVideo('5T-iXNNiwIs', width=500, height=300)
n_inputs = model.classifier[6].in_features

classification_layer = nn.Linear(n_inputs, len(train_set.dataset.classes))

model.classifier[6] = classification_layer
n_inputs, len(train_set.dataset.classes)
for param in model.features.parameters():

    param.requires_grad = False
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Device: {device}')

model.to(device)
def training_loop(loader, epoch):

    

    running_loss = 0.

    running_accuracy = 0.

    

    for i, data in enumerate(loader):

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)



        optimizer.zero_grad()        



        outputs = model(inputs)

        

        loss = criterion(outputs, labels)

        loss.backward()

        

        optimizer.step()



        running_loss += loss.item()





        predicted = torch.argmax(F.softmax(outputs, dim = 1), dim = 1)

        

        

        equals = predicted == labels

        

        

        accuracy = torch.mean(equals.float())

        running_accuracy += accuracy



        if i %50 == 0:

            

            print(f'Epoch: {epoch+1} | loop {i+1}/{len(loader)} Loss: {loss.item():.5f} - Accuracy: {accuracy:.5f}')

        



    print(f'>>>>> Epoch: {epoch+1} Loss: {running_loss/len(loader):.5f} - Accuracy: {running_accuracy/len(loader):.5f}')
%%time 

for epoch in range(2):

    print('Training...')

    training_loop(train_loader, epoch)

    model.eval()

    print('Validation...')

    training_loop(test_loader, epoch)

    model.train()
images, labels = next(iter(test_loader))

model.eval()

predicted = model(images.to(device)).cpu()

predicted = torch.argmax(F.softmax(predicted, dim = 1), dim = 1)

labels, predicted = labels.detach().numpy(), predicted.detach().numpy()
images = images.permute(0, 2, 3, 1).numpy()

images.shape
idx_to_class = {k: v for v, k in test_set.dataset.class_to_idx.items()}

labels = [idx_to_class[label] for label in labels]

predicted = [idx_to_class[label] for label in predicted]



labels, predicted
#Plotting the results.

fig, ax = plt.subplots(6, 5, figsize = (15, 18))

i = 0

for row in range(6):

    ax_row = ax[row]

    for column in range(5):

        ax_column = ax_row[column]

        ax_column.imshow(images[i])

        ax_column.set_xticklabels([])

        ax_column.set_yticklabels([])

        col = 'blue' if labels[i] == predicted[i] else 'red'

        ax_column.set_title(f'Label:{labels[i]}, \nPredicted:{predicted[i]}', color = col)

        i += 1 

        plt.tight_layout()