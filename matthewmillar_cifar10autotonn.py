import os

import numpy as np

import matplotlib.pyplot as plt





import torchvision

from torchvision import transforms

from torchvision import datasets

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SubsetRandomSampler
'''

This method is for counting the number of images per class

params: Data directory that holds multiple photos. The folder name should be the class name for use in ImageGenerator (Kears)

output: It will print the class name and the total number of images per class

'''

def count_images_per_class(directory):

    for folder in os.listdir(directory):

        count = 0

        for image in os.listdir(os.path.join(directory, folder)):

            count +=1

        print("Number of images in Class {} are {}".format(folder, count))

    
# Look at the classes that are avalialbe and load the unedited data

RAW_DATA_DIR = "../input/cifar10-pngs-in-folders/cifar10/"

RAW_TRAIN_DIR = os.path.join(RAW_DATA_DIR, 'train/')

RAW_TEST_DIR = os.path.join(RAW_DATA_DIR, 'test/')

count_images_per_class(RAW_TRAIN_DIR)
# Use cleaned dataset

DATA_DIR = "../input/cleanedcifar10/cifar10_cleaned/cifar10/"

TRAIN_DIR = os.path.join(DATA_DIR, 'train')

TEST_DIR = os.path.join(DATA_DIR, 'test')

# Now count the number of images that remain for each class to ensure a blanced dataset

count_images_per_class(TRAIN_DIR)
#Basic Transforms

SIZE = (32,32) # Resize the image to this shape

# Test and basic transform. This will reshape and then transform the raw image into a tensor for pytorch

basic = transforms.Compose([transforms.Resize(SIZE),

                            transforms.ToTensor()])



# Normalized transforms (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261) retrived from here https://github.com/kuangliu/pytorch-cifar/issues/19

mean = (0.4914, 0.4822, 0.4465) # Mean

std = (0.247, 0.243, 0.261) # Standard deviation

# This will transform the image to the Size and then normalize the image

norm_tran = transforms.Compose([transforms.Resize(SIZE),

                                transforms.ToTensor(), 

                                transforms.Normalize(mean=mean, std=std)])



#Simple Data Augmentation

# Data augmentations

'''

Randomly flip the images both virtically and horizontally this will cover and orientation for images

Randomly rotate the image by 15. This will give images even more orientation than before but with limiting the black board issue of rotations

Random Resie and crop this will resize the image and remove any excess to act like a zoom feature

Normalize each image and make it a tensor

'''

aug_tran = transforms.Compose([transforms.RandomHorizontalFlip(),

                               transforms.RandomRotation(15),

                               transforms.RandomResizedCrop(SIZE, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=3),

                               transforms.ToTensor(),

                               transforms.Normalize(mean=mean, std=std)])
# Create Dataset

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=aug_tran)

test_dataset  = datasets.ImageFolder(TEST_DIR, transform=norm_tran) #No augmentation for testing sets
# Helper Function to show a tensor image in matlibplot.

def imshow(img):

    img = img / 2 + 0.5  # unnormalize

    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
# Data loaders

# Parameters for setting up data loaders

BATCH_SIZE = 32

NUM_WORKERS = 4

VALIDATION_SIZE = 0.15



# Validatiaon split

num_train = len(train_dataset) # Number of training samples

indices = list(range(num_train)) # Create indices for each set

np.random.shuffle(indices) # Randomlly sample each of these by shuffling

split = int(np.floor(VALIDATION_SIZE * num_train)) # Create the split for validation

train_idx , val_idx = indices[split:], indices[:split] # Create the train and validation sets

train_sampler = SubsetRandomSampler(train_idx) # Subsample using pytroch

validation_sampler = SubsetRandomSampler(val_idx) # same here but for validation



# Create the data loaders

train_loader = DataLoader(train_dataset, 

                          batch_size=BATCH_SIZE,

                          sampler=train_sampler, 

                          num_workers=NUM_WORKERS)



validation_loader = DataLoader(train_dataset, 

                               batch_size=BATCH_SIZE,

                               sampler=validation_sampler,

                               num_workers=NUM_WORKERS)



test_loader = DataLoader(test_dataset, 

                         batch_size=BATCH_SIZE, 

                         shuffle=False, 

                         num_workers=NUM_WORKERS)
# Simple visuzlization for dataloaders to check what they are producing. 

train_dataiter = iter(train_loader)

images, labels = train_dataiter.next()

images = images.numpy()

fig = plt.figure(figsize=(25, 4))

# display 20 images

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title([labels[idx]])
# Simple visuzlization for dataloaders to check what they are producing. 

test_dataiter = iter(test_loader)

images, labels = test_dataiter.next()

images = images.numpy()

fig = plt.figure(figsize=(25, 4))

# display 20 images

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title([labels[idx]])


# define the NN architecture

class ConvAutoencoder(nn.Module):

    def __init__(self):

        super(ConvAutoencoder, self).__init__()

        ## encoder layers ##

        # conv layer (depth from 1 --> 16), 3x3 kernels

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  

        # conv layer (depth from 16 --> 4), 3x3 kernels

        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)

        # pooling layer to reduce x-y dims by two; kernel and stride of 2

        self.pool = nn.MaxPool2d(2, 2)

        

        ## decoder layers ##

        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)

        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)





    def forward(self, x):

        ## encode ##

        # add hidden layers with relu activation function

        # and maxpooling after

        x = torch.relu(self.conv1(x))

        x = self.pool(x)

        # add second hidden layer

        x = torch.relu(self.conv2(x))

        x = self.pool(x)  # compressed representation

        

        ## decode ##

        # add transpose conv layers, with relu activation function

        x = torch.relu(self.t_conv1(x))

        # output layer (with sigmoid for scaling from 0 to 1)

        x = torch.sigmoid(self.t_conv2(x))

                

        return x
use_gpu = torch.cuda.is_available()

print("IS there a GPU? : ", use_gpu)
# Build the Net

ae_model = ConvAutoencoder()

print(ae_model)

if use_gpu:

    ae_model.cuda()
# Loss and optimizers

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=3, verbose=True) # Automatically reduce learning rate on plateau
# number of epochs to train the model

n_epochs = 35

ae_model_filename = 'cifar_autoencoder.pt'

train_loss_min = np.Inf # track change in training loss



ae_train_loss_matrix = []

for epoch in range(1, n_epochs+1):

    # monitor training loss

    train_loss = 0.0

    

    ###################

    # train the model #

    ###################

    for data in train_loader:

        # _ stands in for labels, here

        # no need to flatten images

        

        images, _ = data

        if use_gpu:

            images = images.cuda()

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        outputs = ae_model(images)

        # calculate the loss

        loss = loss_function(outputs, images)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update running training loss

        train_loss += loss.item()*images.size(0)

            

    # print avg training statistics 

    train_loss = train_loss/len(train_loader)

    scheduler.step(train_loss)

    ae_train_loss_matrix.append([train_loss, epoch])

    

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    

    # save model if validation loss has decreased

    if train_loss <= train_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        train_loss_min,

        train_loss))

        torch.save(ae_model.state_dict(), ae_model_filename)

        train_loss_min = train_loss
# For plotting of the loss for the autoencoder

x, y = [], []

for i in range(len(ae_train_loss_matrix)):

    x.append(ae_train_loss_matrix[i][0])

    y.append(ae_train_loss_matrix[i][1])
#plt.scatter(ae_train_loss_matrix[0],ae_train_loss_matrix[1])

plt.plot(y, x)

plt.title("AE Validation Loss")

plt.show()
# Load the best model with lowest loss

ae_model.load_state_dict(torch.load(ae_model_filename))
#Check how good the autoencoder is

# Check results

dataiter = iter(test_loader)

images, labels = dataiter.next()

if use_gpu:

    images = images.cuda()

# Get sample outputs

output = ae_model(images)

# Prep images for display

images = images.cpu()

images = images.numpy()

print(images.shape)

# use detach when it's an output that requires_grad

output = output.cpu()

output = output.detach().numpy()

# plot the first ten input images and then reconstructed images

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

# input images on top row, reconstructions on bottom

for images, row in zip([images, output], axes):

    for img, ax in zip(images, row):

        img = img / 2 + 0.5  # unnormalize

        ax.imshow(np.transpose(img, (1, 2, 0))) 

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)
for child in ae_model.children():

    print(child)

    

class MyModel(nn.Module):

    def __init__(self):

        super(MyModel, self).__init__()

        image_modules = list(ae_model.children())[:-2] #get only the encoder layers

        self.modelA = nn.Sequential(*image_modules)

        # Shape of max pool = 4, 112, 112

        self.fc1 = nn.Linear(4*16*16, 1024)

        self.fc2 = nn.Linear(1024,512)

        self.out = nn.Linear(512, 10)

        

        self.drop = nn.Dropout(0.2)

        

    def forward(self, x):

        x = self.modelA(x)

        x = x.view(x.size(0),4*16*16)

        x = torch.relu(self.fc1(x))

        x = self.drop(x)

        x = torch.relu(self.fc2(x))

        x = self.drop(x)

        x = self.out(x)

        return x
model = MyModel()

print(model)
def calc_accuracy(output,labels):

    max_vals, max_indices = torch.max(output,1)

    acc = (max_indices == labels).sum().item()/max_indices.size()[0]

    return acc
#Freze the autoencoder layers so they do not train. We did that already

# Train only the linear layers

for child in model.children():

    if isinstance(child, nn.Linear):

        print("Setting Layer {} to be trainable".format(child))

        for param in child.parameters():

            param.requires_grad = True

    else:

        for param in child.parameters():

            param.requires_grad = False

# Optimizer and Loss function

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-05)

# Decay LR by a factor of 0.1 every 7 epochs

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=3, verbose=True)
if use_gpu:

    model.cuda()

    

model_filename = 'model_cifar10.pt'

n_epochs = 40

valid_loss_min = np.Inf # track change in validation loss

train_loss_matrix = []

val_loss_matrix = []

val_acc_matrix = []



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    valid_loss = 0.0

    

    train_correct = 0

    train_total = 0

    

    val_correct = 0

    val_total = 0

    

    

    ###################

    # train the model #

    ###################

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        # move tensors to GPU if CUDA is available

        if use_gpu:

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

        train_loss += loss.item()*data.size(0)

        

        

    ######################    

    # validate the model #

    ######################

    model.eval()

    val_acc = 0.0

    for batch_idx, (data, target) in enumerate(validation_loader):

        # move tensors to GPU if CUDA is available

        if use_gpu:

            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        loss = criterion(output, target)

        # update average validation loss 

        valid_loss += loss.item()*data.size(0)

        

        val_acc += calc_accuracy(output, target)

        

        

    

    # calculate average losses

    train_loss = train_loss/len(train_loader.sampler)

    valid_loss = valid_loss/len(validation_loader.sampler)

    #exp_lr_scheduler.step()

    scheduler.step(valid_loss)

    

    # Add losses and acc to plot latter

    train_loss_matrix.append([train_loss, epoch])

    val_loss_matrix.append([valid_loss, epoch])

    val_acc_matrix.append([val_acc, epoch])

        

    # print training/validation statistics 

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\tValidation Accuracy: {:.6f}'.format(

        epoch, train_loss, valid_loss, val_acc))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,valid_loss))

        

        torch.save(model.state_dict(), model_filename)

        valid_loss_min = valid_loss
val_acc_x, val_loss_x, train_loss_x = [],[],[]

y = []

for i in range(len(train_loss_matrix)):

    train_loss_x.append(train_loss_matrix[i][0])

    y.append(train_loss_matrix[i][1])

for i in range(len(val_loss_matrix)):

    val_loss_x.append(val_loss_matrix[i][0])

for i in range(len(val_acc_matrix)):

    val_acc_x.append(val_acc_matrix[i][0])

    

print(len(val_acc_x), len(train_loss_x), len(val_loss_x), len(y))
plt.plot(y, val_acc_x)

plt.title("Validation Accuracy")

plt.show()

plt.plot(y, train_loss_x)

plt.plot(y, val_loss_x)

plt.title("Train/Validation Loss")

plt.show()
# Load model with lowest validation loss

model.load_state_dict(torch.load(model_filename))
# specify the image classes

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',

           'dog', 'frog', 'horse', 'ship', 'truck']
# track test loss

test_loss = 0.0

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



model.eval()

# iterate over test data

for batch_idx, (data, target) in enumerate(test_loader):

    # move tensors to GPU if CUDA is available

    if use_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the batch loss

    loss = criterion(output, target)

    # update test loss 

    test_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)    

    # compare predictions to true label

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not use_gpu else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class



    for i in range(16):

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



# average test loss

test_loss = test_loss/len(test_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(10):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
