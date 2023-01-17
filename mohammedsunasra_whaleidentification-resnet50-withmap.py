import os

import json

import pandas as pd

import numpy as np

import time

import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image

from imageio import imread

sns.set_style("darkgrid")

%matplotlib inline
import warnings

warnings.filterwarnings('ignore')
import torch

import torchvision

from torch import nn

from torch import optim

from torch.optim import lr_scheduler

from torchvision import datasets, transforms, models

from torch.utils.data import Dataset

from torch.utils.data.sampler import SubsetRandomSampler
gpu = torch.cuda.is_available()

if gpu:

    print("GPU available")

else:

    print("GPU NOT available! Training will happen on CPU")
DATA_PATH = "../input/"
df = pd.read_csv(DATA_PATH + 'train.csv')

df.head()
n_classes = df.Id.unique()

class_to_idx = {class_name:idx for idx, class_name in enumerate(n_classes)}

idx_to_classes = {idx:class_name for class_name, idx in class_to_idx.items()}
train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(),

    #transforms.RandomGrayscale(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

])
class WhaleDataset(Dataset):

    """

    Dataset to generate batches of multiple images and labels from a CSV file.

    Purpose: To work with CSV files where the format is (file_name, cclass_label)

    and generate batches of data(images, labels) on-the-fly.

    """

    def __init__(self, csv_file_path, image_path, image_size, class_to_idx, transform=None):

        self.data = pd.read_csv(csv_file_path)

        self.image_path = image_path

        self.transform = transform

        self.class_to_idx = class_to_idx



    def __len__(self):

        """

        Returns the no of datapoints in the dataset

        """

        return len(self.data)

    

    def __getitem__(self, index):

        """

        Returns a batch of data given an index

        """

        image_name = self.data.iloc[index, 0]

        image = Image.open(self.image_path + image_name)

        image = image.convert('RGB')

        image = image.resize(image_size, Image.ANTIALIAS) 

        if self.transform is not None:

            image = self.transform(image)

        label = self.data.iloc[index, 1]

        label = self.class_to_idx[label]

        label = torch.from_numpy(np.asarray(label))

        

        return image, label
#path of the csv file containing info about images and labels

CSV_PATH = DATA_PATH + 'train.csv'

#path where the actual training images are stored

IMAGE_PATH = DATA_PATH + 'train/'

#no of images we want to display while plotting

n_images = 10

#image size in width * height

image_size = (224,224)
whale_dataset = WhaleDataset(CSV_PATH, IMAGE_PATH, image_size, class_to_idx, transform=train_transform)
valid_size = 0.2
no_train = len(whale_dataset)

indices = list(range(no_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * no_train))

train_indices, valid_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(valid_indices)
train_loader = torch.utils.data.DataLoader(whale_dataset, batch_size=32, sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(whale_dataset, batch_size=32, sampler=valid_sampler)
images, labels = next(iter(train_loader))

#images = images.numpy()
def display_image(inp, title=None):

    inp = inp.numpy()

    inp = np.transpose(inp, (1,2,0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = inp * std + mean

    inp = np.clip(inp,0,1)

    if title is not None:

        plt.title(title)

    plt.figure(figsize=(32,6))

    plt.imshow(inp)

    plt.pause(0.001)
out = torchvision.utils.make_grid(images, nrow=8, padding=0)

#display_image(out, title=[idx_to_classes[x.item()] for x in labels])

display_image(out, title=None)
model = models.resnet50(pretrained=True)

print(model)
for param in model.parameters():

    param.requires_grad = False
input_size = model.fc.in_features

output_size = model.fc.out_features

print(input_size)

print(output_size)
last_fc = nn.Sequential(

    nn.Linear(input_size, len(class_to_idx)),nn.Softmax(dim=1))
model.fc = last_fc

print(model)
if gpu:

    model.cuda()
# def prepare_dataframe(index, labels):

#     index = index.cpu().numpy()

#     labels = labels.cpu().numpy()

#     df_index = pd.DataFrame(index)

#     df_labels = pd.DataFrame(labels, columns=['target'])

#     df = pd.concat([df_index,df_labels ], axis=1)

#     df['precision'] = 0

#     del df_index

#     del df_labels

#     return df
def map_per_image(label, predictions):

    label = int(label.cpu().numpy())

    predictions = list(predictions.cpu().numpy())

    try:

        return 1 / (predictions.index(label) + 1)

    except ValueError:

        return 0.0
def map_per_batch(labels, predictions):

    return np.mean([map_per_image(l,p) for l,p in zip(labels, predictions)])
# def get_precision(x):

#     target = np.array(x[-2])

#     labels = np.array(x[:-2])

    

#     target_position, = np.where(labels == target)

    

#     if len(target_position) > 0:

#         target_position = target_position[0] + 1

#         precision = (1 / target_position)

#     else:

#         precision = 0

#     return precision
optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

criterion = nn.CrossEntropyLoss()

scheduler = lr_scheduler.StepLR(optimizer,gamma=0.1, step_size=10)
n_epochs = 30

min_loss = np.Inf
training_losses = []

valid_losses = []

min_valid_loss = np.Inf
for e in range(1, n_epochs+1):

    print(f"-----------Epoch {e}/{n_epochs}-------------------")

    #switching the model to training mode

    model.train()

    #initialising starting values for training and validation loss

    training_loss = 0.0

    validation_loss = 0.0

    #Initializing starting values for training accuracy

    total_train = 0

    total_correct_train = 0

    #Initializing starting values for validation accuracy

    total_validation = 0

    total_correct_validation = 0

    train_precision = 0

    valid_precision = 0

    for images, labels in train_loader:

        if gpu: #move the data to GPU if available

            images, labels = images.cuda(), labels.cuda()

        #clearing out gradients

        optimizer.zero_grad()

        #doing the forward pass

        output = model(images)

        # Calculating training accuracy

        # Total number of labels

        total_train += len(labels)

        # Getting predicted labels

        predicted = torch.max(output.data, 1)[1]        

        # Total correct predictions

        total_correct_train += torch.sum(predicted == labels).item()

        #Calculating precision

        probs, predictions = output.topk(5, dim=1)

        batch_precision = map_per_batch(labels, predictions)

        #print(f"Batch Precision is {batch_precision}")

        #df = prepare_dataframe(index, labels)

        #df['precision'] = df.apply(get_precision, axis=1)

        #batch_precision = df['precision'].sum()

        train_precision += batch_precision

        

        #calculating the loss from the forward pass

        loss = criterion(output, labels)

        #propagating the error backwards

        loss.backward()

        #updating weights and biases

        optimizer.step()

        #adding training loss for a batch

        training_loss += loss.item() * images.size(0)



    #switching the model to evaluation mode

    model.eval()

    for images, labels in valid_loader:

        with torch.no_grad():

            if gpu: #move the data to GPU if available

                images, labels = images.cuda(), labels.cuda()

            #doing the forward pass for validation images

            output = model(images)

            

            #calculating valdation accuracy

            total_validation += len(images)

            predicted = torch.max(output.data, 1)[1]

            total_correct_validation += torch.sum(predicted == labels).item()

            #calculating CE loss for validation images

            loss = criterion(output, labels)

            #adding up validation loss for a batch

            validation_loss += loss.item() * images.size(0)



    train_loss = training_loss/len(train_loader.dataset)

    valid_loss = validation_loss/len(valid_loader.dataset)

    print(f"Total training precision is {train_precision}")

    avg_train_precision = train_precision / float(len(train_loader))

    

    train_accuracy = (total_correct_train / float(total_train)) * 100

    valid_accuracy = (total_correct_validation / float(total_validation)) * 100

    print(f"Mean Average precision for training is {avg_train_precision}")

    print(f"Training loss for epoch no {e} is {train_loss}")

    print(f"Training accuracy for epoch no {e} is {train_accuracy}")

    print(f"Validation loss for epoch no {e} is {valid_loss}")

    print(f"Validation accuracy for epoch no {e} is {valid_accuracy}")



    training_losses.append(train_loss)

    valid_losses.append(valid_loss)



    if valid_loss <= min_valid_loss:

        print(f"Validation loss decreased from {min_valid_loss} to {valid_loss}.....Saving model.....")

        #torch.save(model.state_dict(), 'model_whale_resnet50.pt')

        torch.save({

            'epoch': e,

            'model_state_dict': model.state_dict(),

            'optimizer_state_dict': optimizer.state_dict(),

            'val_loss': valid_loss,

            'train_loss': train_loss

            }, 'model_resnet50_parameters.tar')

        min_valid_loss = valid_loss

        

    scheduler.step()