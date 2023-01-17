import os
import json
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
%matplotlib inline
import torch
import torchvision
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import torch.nn.functional as F
gpu = torch.cuda.is_available()
if not gpu:
    print("CUDA Not available: Training will happen on CPU")
else:
    print("CUDA is available: Training will happen on GPU")
PATH = '../input/flower_data/flower_data'
batch_size = 64
transform_dict = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
}
train_dataset = datasets.ImageFolder(PATH + '/train', transform=transform_dict['train'])
valid_dataset = datasets.ImageFolder(PATH + '/valid', transform=transform_dict['train'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,shuffle=True)
print("Training set statistics: ")
print("Total no of datapoints in the training set are", len(train_loader.dataset))
print("Total no of batches in the training set are", len(train_loader))

print("Validation set statistics: ")
print("Total no of datapoints in the validation set are", len(valid_loader.dataset))
print("Total no of batches in the validation set are", len(valid_loader))
cat_names = open(f'{PATH}/cat_names.json', 'r').read()
cat_names_json = json.loads(cat_names)

print("Total No of output classes are: ", len(cat_names_json))
model = models.vgg19(pretrained=True)
model
for param in model.features.parameters():
    param.requires_grad = False
input_size = model.classifier[6].in_features
last_fc = nn.Linear(input_size, len(cat_names_json))
model.classifier[6] = last_fc
print(model)
n_epochs = 50
min_valid_loss = np.Inf

#losses that we would be using after every epoch
training_losses = []
valid_losses = []
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.1)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
if gpu:
    model.cuda()
else:
    model.cpu()
for e in range(n_epochs):
    print(f"-----------Starting Epoch No {e+1}-------------------")
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
    for images, labels in train_loader:
        if gpu: #move the data to GPU if available
            images, labels = images.cuda(), labels.cuda()
        #clearing out gradients
        optimizer.zero_grad()
        #doing the forward pass
        output = model(images)
        # Calculating training accuracy
        # Total number of labels
        total_train += len(images)
        # Getting predicted labels
        predicted = torch.max(output.data, 1)[1]        
        # Total correct predictions
        total_correct_train += torch.sum(predicted == labels).item()
        
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
    
    train_accuracy = (total_correct_train / float(total_train)) * 100
    valid_accuracy = (total_correct_validation / float(total_validation)) * 100
    print(f"Training loss for epoch no {e+1} is {train_loss}")
    print(f"Training accuracy for epoch no {e+1} is {train_accuracy}")
    print(f"Validation loss for epoch no {e+1} is {valid_loss}")
    print(f"Validation accuracy for epoch no {e+1} is {valid_accuracy}")

    training_losses.append(train_loss)
    valid_losses.append(valid_loss)

    if valid_loss <= min_valid_loss:
        print(f"Validation loss decreased from {min_valid_loss} to {valid_loss}.....Saving model.....")
        torch.save(model.state_dict(), '../../kaggle/models/model_vgg19.pt')
        min_valid_loss = valid_loss
        
    scheduler.step()
x = range(n_epochs)
plt.figure(figsize=(12,6))
#plt.subplot(1,2,0)
plt.plot(x, training_losses)
plt.title("Training loss across all epochs")
plt.xlabel("No of epochs")
plt.ylabel("Training Loss")

#plt.subplot(1,2,1)
plt.figure(figsize=(12,6))
plt.plot(x, valid_losses)
plt.title("Validation loss across all epochs")
plt.xlabel("No of epochs")
plt.ylabel("Validation Loss")

plt.show()
