import numpy as np
import pandas as pd

# Porject modules
import os
print(os.listdir("../input"))
from collections import OrderedDict
import time

# PyTorch modules
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
# Useful project constants
BATCH_SIZE = 24
TYPES_OF_DATASETS = ['train', 'valid'] # Order matters her for train_model function
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../input/flower_data/flower_data/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in TYPES_OF_DATASETS}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=False)
              for x in TYPES_OF_DATASETS}
dataset_sizes = {x: len(image_datasets[x]) for x in TYPES_OF_DATASETS}
class_names = image_datasets['train'].classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.densenet161(pretrained=True)
NUM_CLASSES = len(class_names)
IN_SIZE = model.classifier.in_features  # Expected in_features for desnsenet161 model
NUM_EPOCHS = 10

# Freeze feature block since we're using this model for feature extraction
for param in model.features.parameters():
    param.requires_grad = False
    
# Prep for model training
criterion = nn.CrossEntropyLoss()
# only classifier parameters are being optimized the rest are frozen
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# Fit data to model
def fit(model, data_loader, criterion, optimizer=None, train=False):
    if train:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        if train:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
        # update running training loss
        running_loss += loss.item() * data.size(0)

        # Calculate accuracy
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        correct = top_class == target.view(*top_class.shape)
        running_acc += torch.mean(correct.type(torch.FloatTensor))
        
    loss = running_loss / len(data_loader.dataset)
    acc = 100. * running_acc / len(data_loader)
    
    return loss, acc
# Function for initiating model training
def train_model(model, train_loader, valid_loader, criterion, optimizer,
               scheduler, num_epochs=10):
    """
    Trains and validates the data using the specified model and parameters.
    """
    model.to(device)

    start_train_timer = time.time()
    for epoch in range(num_epochs):
        # Start timer
        start = time.time()
        # Pass forward through the model
        scheduler.step()
        train_loss, train_accuracy = fit(model, train_loader, criterion=criterion,
                                         optimizer=optimizer, train=True)
        valid_loss, valid_accuracy = fit(model, valid_loader, criterion=criterion,
                                         train=False)

        # calculate average loss over an epoch
        elapshed_epoch = time.time() - start

        # print training/validation statistics 
        print('Epoch: {} - completed in: {:.0f}m {:.0f}s'.format(
            epoch + 1, elapshed_epoch // 60, elapshed_epoch % 60))
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            train_loss, valid_loss))
        print('\tTraining accuracy: {:.3f} \tValidation accuracy: {:.3f}'.format(
            train_accuracy, valid_accuracy))
            
    training_time = time.time() - start_train_timer
    hours = training_time // (60 * 60)
    training_time -= hours * 60 * 60
    print('Model training completed in: {:.0f}h {:.0f}m {:.0f}s'.format(
            hours, training_time // 60, training_time % 60))
# Get dataloaders for training and validation sets
train_loader = dataloaders['train']
valid_loader = dataloaders['valid']
%%time
# Run training model without precalculating frozen features
train_model(model, train_loader, valid_loader, criterion,
            optimizer, scheduler, num_epochs=NUM_EPOCHS)
# Function for generating convoluted features and labels for given dataset and model
def preconvfeat(dataloader, model):
    model.to(device)
    conv_features = []
    labels_list = []
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.features(inputs)  # calculate values for features block only
        conv_features.extend(output.data.cpu()) # save to CPU since it has much larger RAM
        labels_list.extend(labels.data.cpu())

    return (conv_features, labels_list)
# Convoluated feature dataset class for retrieving datasets
class ConvDataset(Dataset):
    def __init__(self, feats, labels):
        self.conv_feats = feats
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.conv_feats[idx], self.labels[idx]
# Custom fc class for densenet161 that uses precalculated feature values
class FullyConnectedModel(nn.Module):

    def __init__(self,in_size,out_size):
        super().__init__()
        self.fc = nn.Linear(in_size,out_size)

    def forward(self, features):
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.fc(out)
        return out

fc = FullyConnectedModel(IN_SIZE, NUM_CLASSES)
%%time
# Precalculate feature values for the training and validation data
conv_train_data, labels_train_data = preconvfeat(dataloaders['train'], model)
conv_valid_data, labels_valid_data = preconvfeat(dataloaders['valid'], model)
# Convert the calculated data into Dataset opbject
feat_train_dataset = ConvDataset(conv_train_data, labels_train_data)
feat_valid_dataset = ConvDataset(conv_valid_data, labels_valid_data)
# Generate DataLoader objects for calculated Datasets
train_dataloader = DataLoader(feat_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(feat_valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

# run training model
train_model(fc, train_dataloader, valid_dataloader, criterion,
            optimizer, scheduler, num_epochs=NUM_EPOCHS)