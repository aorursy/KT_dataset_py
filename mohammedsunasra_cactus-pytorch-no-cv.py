#python specific modules

import os

import time

import copy

import pathlib

from collections import defaultdict

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from PIL import Image
#pytorch specific modules

import torch

import torchvision

import torch.nn as nn

import torch.optim as optmim

from torch.optim import lr_scheduler

from torchvision import models, datasets, transforms

from torch.utils.data import Dataset

from torch.utils.data.sampler import SubsetRandomSampler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_path = pathlib.Path("../input/")
df_data = pd.read_csv(base_path/'train.csv')
#Custom data generator class

class CactusDataset(Dataset):

    """

    Dataset to generate batches of multiple images and labels from a CSV file.

    Purpose: To work with CSV files where the format is (file_name, cclass_label)

    and generate batches of data(images, labels) on-the-fly.

    """

    def __init__(self, df_data, image_path, image_size, transform=None):

        self.data = df_data

        self.image_path = image_path

        self.transform = transform

        

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

        image = Image.open(str(self.image_path) + '/' +image_name)

        image = image.convert('RGB')

        image = image.resize(image_size, Image.ANTIALIAS) 

        if self.transform is not None:

            image = self.transform(image)

        label = self.data.iloc[index, 1]

        label = torch.from_numpy(np.asarray(label))

        

        return image, label
train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(),

    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),

    transforms.RandomAffine(0.1),

    transforms.RandomGrayscale(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

])
valid_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

])
image_path = base_path/'train'/'train'

image_size = (224, 224)

bs = 64
cac_dataset = CactusDataset(df_data, image_path, image_size, transform=train_transform)
sample_loader = torch.utils.data.DataLoader(cac_dataset, batch_size=8, shuffle=True)

images, labels = next(iter(sample_loader))
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

display_image(out, title=None)

print(labels)
X = df_data['id']

y = df_data['has_cactus']
X_train, X_val, y_train, y_val = train_test_split(X, y,

                                                stratify=y, 

                                                test_size=0.20, shuffle=True)
df_train = pd.DataFrame({'id': X_train.values, 'has_cactus':y_train.values})

df_val = pd.DataFrame({'id':X_val.values, 'has_cactus':y_val.values})
cac_dataset_train = CactusDataset(df_train, image_path, image_size, transform=train_transform)

cac_dataset_valid = CactusDataset(df_val, image_path, image_size, transform=valid_transform)
train_loader = torch.utils.data.DataLoader(cac_dataset_train, batch_size=32, shuffle=True)

valid_loader = torch.utils.data.DataLoader(cac_dataset_valid, batch_size=32, shuffle=True)
def get_base_model(model_name, num_classes, pretrained=True, unfreeze=True, to_gpu=True):

    if model_name == "resnet":

        model = models.resnet152(pretrained=pretrained)

        input_features = model.fc.in_features

        fc_custom = nn.Linear(input_features, num_classes)

        model.fc = fc_custom

    

    elif model_name == "densenet":

        model = models.densenet121(pretrained=pretrained)

        input_features = model.classifier.in_features

        fc_custom = nn.Linear(input_features, num_classes)

        model.classifier = fc_custom

    

    if unfreeze:

        for param in model.parameters():

            param.requires_grad = True

    

    if to_gpu:

        model = model.to(device)

    return model
model = get_base_model("resnet", 2)
model
criterion = nn.CrossEntropyLoss()

optimizer = optmim.SGD([

            {'params': model.layer1.parameters(), 'lr': 1e-6},

            {'params': model.layer2.parameters(), 'lr': 1e-5},

            {'params': model.layer3.parameters(), 'lr': 1e-4},

            {'params': model.layer4.parameters(), 'lr': 1e-4},

            {'params': model.fc.parameters(), 'lr': 1e-3}

        ], lr=1e-3)

#optimizer = optmim.SGD(model.parameters(), lr=1e-7, momentum=0.9)

#scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
def get_auc_score(y_true, y_pred):

    y_true = np.array([item for sublist in y_true for item in sublist])

    y_pred = np.array([item for sublist in y_pred for item in sublist])

    return roc_auc_score(y_true, y_pred)
n_epochs = 50
def train_model(model, dataloaders, criterion, optimizer, epochs=50):

    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    best_loss = np.Inf

    metrics = defaultdict(list)

    losses = defaultdict(list)

    

    for epoch_no in range(epochs):

        print("*" * 100)

        print(f"Starting epoch no {epoch_no+1}")

        for phase in ['train','valid']:

            y_true = list()

            y_pred = list()

            

            if phase == 'train':

                model.train()

            else:

                model.eval()

            

            running_loss = 0.0

            running_corrects = 0

            

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)

                

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    idxs, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                

                probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]

                

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

                

                y_true.append(list(labels.data.cpu().numpy()))

                y_pred.append(list(probs.detach().cpu().numpy()))

            

            auc_score = get_auc_score(y_true, y_pred)

            

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)

                

            if phase == 'valid' and epoch_loss < best_loss:

                print(f"Validation loss decreased from {best_loss} to {epoch_loss}. Saving Model ")

                #best_acc = epoch_acc

                best_loss = epoch_loss

                checkpoint = {'model': model,

                              'state_dict': model.state_dict(),

                              'optimizer' : optimizer.state_dict()

                }

                torch.save(checkpoint, 'model_resnet_50_v2.pth')

                #torch.save(model.state_dict(), "model_resnet_50_v1.0.pt")

                best_model_wts = copy.deepcopy(model.state_dict())

                

            print(f"Ending epoch no {epoch_no+1} with below stats")

            print(f"-------------Stats for {phase}-------------")

            print(f"Loss: {epoch_loss}... Accuracy: {epoch_acc}")

            print(f"AUC for {phase} is {auc_score}")

            

            metrics[phase].append(epoch_acc)

            losses[phase].append(epoch_loss)

            

    time_elapsed = time.time() - start_time

    print(f"Total time taken in training: {time_elapsed / 60} minutes")

    

    # load best model weights

    model.load_state_dict(best_model_wts)

    return model, metrics, losses

    
def plot_metrics_with_loss(metrics, losses):

    train_metrics = metrics['train']

    valid_metrics = metrics['valid']

    

    train_loss = losses['train']

    valid_loss = losses['valid']

    

    x = list(range(1, n_epochs+1))

    

    fig = plt.figure(figsize=(15,10))

    

    plt.subplot(2, 2, 1)

    plt.title('Training Loss Graph over multiple epochs')

    plt.plot(x, train_loss)



    plt.subplot(2, 2, 2)

    plt.title('Validation Loss Graph over multiple epochs')

    plt.plot(x, valid_loss)



    plt.subplot(2, 2, 3)

    plt.title('Training Metrics(Accuracy) Graph over multiple epochs')

    plt.plot(x, train_metrics)



    plt.subplot(2, 2, 4)

    plt.title('Validation Metrics(Accuracy) Graph over multiple epochs')

    plt.plot(x, valid_metrics)



    plt.show()
dataloaders = {

        'train': train_loader,

        'valid': valid_loader

    }

model, metrics, losses = train_model(model, dataloaders, criterion, optimizer)

plot_metrics_with_loss(metrics, losses)
df_submission = pd.read_csv(base_path/'sample_submission.csv')
def load_model(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)

    model = checkpoint['model']

    model.load_state_dict(checkpoint['state_dict'])

    model.cpu()

    for parameter in model.parameters():

        parameter.requires_grad = False



    model.eval()

    return model
model = load_model('model_resnet_50_v2.pth')
test_image_path = base_path / 'test' / 'test'
def add_full_path(file_name):

    full_path = str(test_image_path) + '/' + file_name

    return full_path
df_submission['id'] = df_submission['id'].apply(add_full_path)
def read_image(image_path):

    image = Image.open(image_path)

    image = image.convert('RGB')

    image = image.resize(image_size, Image.ANTIALIAS)

    return image
test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

])
def get_predictions(image_path):

    image = read_image(image_path)

    image = test_transform(image)

    image = torch.unsqueeze(image, 0)

    pred = model(image)

    p_cactus = torch.nn.functional.softmax(pred, dim=1)[0][1].item()

    return p_cactus

df_submission['has_cactus'] = df_submission['id'].apply(get_predictions)
df_submission.head()
def remove_full_path(image_path):

    paths = image_path.split("/")

    path = paths[4]

    return path
df_submission['id'] = df_submission['id'].apply(remove_full_path)
df_submission.head()
df_submission.to_csv('submission_resnet50_unfreeze_1.csv', index=None)