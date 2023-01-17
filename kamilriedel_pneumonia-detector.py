import numpy as np
import pandas as pd 
import os
from glob import glob
# loading the directories 
trai_dir = '../input/chest-xray-pneumonia/chest_xray/train/'
val_dir = '../input/chest-xray-pneumonia/chest_xray/val/'
test_dir = '../input/chest-xray-pneumonia/chest_xray/test/'

# getting the number of classes
folders = glob(trai_dir + '/*')
num_classes = len(folders)
class_labels=os.listdir(trai_dir)
print ('Total number of classes = ' + str(num_classes))
print('Class names: {0}'.format(class_labels))
# getting number of files
train_files = np.array(glob(trai_dir+"*/*"))
val_files = np.array(glob(val_dir+"*/*"))
test_files = np.array(glob(test_dir+"*/*"))

# print number of images in each dataset
print('There are %d total train images.' % len(train_files))
print('There are %d total validation images.' % len(val_files))
print('There are %d total test images.' % len(test_files))
train_normal = np.array(glob(trai_dir+"NORMAL/*"))
val_normal = np.array(glob(val_dir+"NORMAL/*"))
test_normal = np.array(glob(test_dir+"NORMAL/*"))
train_pneumonia = np.array(glob(trai_dir+"PNEUMONIA/*"))
val_pneumonia = np.array(glob(val_dir+"PNEUMONIA/*"))
test_pneumonia = np.array(glob(test_dir+"PNEUMONIA/*"))

print('There are %d total normal train images.' % len(train_normal))
print('There are %d total normal validation images.' % len(val_normal))
print('There are %d total normal test images.' % len(test_normal))
print('There are %d total pneumonia train images.' % len(train_pneumonia))
print('There are %d total pneumonia validation images.' % len(val_pneumonia))
print('There are %d total pneumonia test images.' % len(test_pneumonia))
import cv2                
import matplotlib.pyplot as plt   
import matplotlib.image as mpimg

file_path = np.concatenate((train_normal[0:3],train_pneumonia[0:3]))

fig = plt.figure(figsize=(13, 9))

for i in range(len(file_path)):
    ax = fig.add_subplot(2,3,i+1, xticks=[], yticks=[])
    img=mpimg.imread(file_path[i])
    ax.set_title(file_path[i].split('/')[-2])
    imgplot = plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
import torch
import torchvision.transforms as transforms
from torchvision import datasets

# Create training and test dataloaders

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32
# resize the picture
size = 256

data_transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(), # randomly flip and rotate
                transforms.RandomRotation(20),
                transforms.Resize(size),
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))]) 
    
data_transform_test = transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(224), 
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), 
                                         (0.229, 0.224, 0.225))]) 

dog_image_dir = '/data/dog_images'

# choose the training and test datasets
train_data = datasets.ImageFolder(trai_dir, transform=data_transform_train)
valid_data = datasets.ImageFolder(val_dir, transform=data_transform_train)
test_data = datasets.ImageFolder(test_dir, transform=data_transform_test)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
data_loader = dict(train=train_loader, valid=valid_loader, test=test_loader)

# Let's verify that the data was loaded correctly by printing out data stats
print('Num training images: ', len(train_data))
print('Num validation images: ', len(valid_data))
print('Num test images: ', len(test_data))
# Visualize some sample train data

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# denormalize the image
def denormalise(image):
    image = np.transpose(image, (1, 2, 0))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image * std + mean).clip(0, 1)
    return image

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 9))
no_vis_imag = 14
for idx in np.arange(no_vis_imag):
    ax = fig.add_subplot(2, no_vis_imag/2, idx+1, xticks=[], yticks=[])
    plt.imshow(denormalise(images[idx]))
    ax.set_title(class_labels[labels[idx]])
import torchvision.models as models
#import torch.nn.functional as F
import torch.nn as nn

# Load the pretrained model from pytorch
VGG16 = models.vgg16(pretrained=True)

# Modify the last layer
n_inputs = VGG16.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(class_labels))

VGG16.classifier[6] = last_layer
## Specify Loss Function and Optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(VGG16.parameters(), lr=0.01)

# check if CUDA is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    VGG16 = VGG16.cuda()
    print('CUDA is available!  Training on GPU ...')
else:
    print('CUDA is not available.  Training on CPU ...')
#from PIL import ImageFile

#ImageFile.LOAD_TRUNCATED_IMAGES = True
        
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model with the possibility to resume analysis and load validation parameters"""

    valid_loss_min = np.Inf # track change in validation loss

    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        # set model to train mode by using dropout to prevent overfitting
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            #train_loss += loss.item()*data.size(0)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    # return trained model
    return model
# train the model
n_epoch = 8
model = train(n_epoch, data_loader, VGG16, optimizer, 
                      criterion, use_cuda, 'model.pt')

# load the model that got the best validation accuracy
model.load_state_dict(torch.load('model.pt'))
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    #for batch_idx, (data, target) in enumerate(loaders['test']):
    for batch_idx, (data, target) in enumerate(test_loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(data_loader, model, criterion, use_cuda)
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
#images = images.numpy() # convert images to numpy for display
images_np = images.cpu().numpy() if not use_cuda else images.numpy()

# move model inputs to cuda, if GPU available
if use_cuda:
    images = images.cuda()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())
#preds = np.squeeze(preds_tensor.cpu().numpy()) # if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))

no_vis_imag = 16
for idx in np.arange(no_vis_imag):
    ax = fig.add_subplot(2, no_vis_imag/2, idx+1, xticks=[], yticks=[])
    plt.imshow(denormalise(images_np[idx]))
    ax.set_title(class_labels[labels[idx]])
    ax.set_title("{} ({})".format(class_labels[preds[idx]], class_labels[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
#images = images.numpy() # convert images to numpy for display
images_np = images.cpu().numpy() if not use_cuda else images.numpy()

# move model inputs to cuda, if GPU available
if use_cuda:
    images = images.cuda()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())
#preds = np.squeeze(preds_tensor.cpu().numpy()) # if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))

no_vis_imag = 12
for idx in np.arange(no_vis_imag):
    ax = fig.add_subplot(2, no_vis_imag/2, idx+1, xticks=[], yticks=[])
    plt.imshow(denormalise(images_np[idx]))
    ax.set_title(class_labels[labels[idx]])
    ax.set_title("{} ({})".format(class_labels[preds[idx]], class_labels[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
