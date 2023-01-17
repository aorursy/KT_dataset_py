#import important libraries

import matplotlib.pyplot as plt

import numpy as np

from torch import nn, optim

from torch.autograd import Variable

import json

import os

import torch

import torch.nn.functional as F

from torchvision import datasets, models, transforms

from torch.utils.data import DataLoader

from collections import OrderedDict

from PIL import Image

from torch import Tensor

from importlib import reload

#reload(helper)

from glob import glob
#Make helper utility
def test_network(net, trainloader):



    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)



    dataiter = iter(trainloader)

    images, labels = dataiter.next()



    # Create Variables for the inputs and targets

    inputs = Variable(images)

    targets = Variable(images)



    # Clear the gradients from all Variables

    optimizer.zero_grad()



    # Forward pass, then backward pass, then update weights

    output = net.forward(inputs)

    loss = criterion(output, targets)

    loss.backward()

    optimizer.step()



    return True
def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))



    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')



    return ax
def view_recon(img, recon):

    ''' Function for displaying an image (as a PyTorch Tensor) and its

        reconstruction also a PyTorch Tensor

    '''



    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

    axes[0].imshow(img.numpy().squeeze())

    axes[1].imshow(recon.data.numpy().squeeze())

    for ax in axes:

        ax.axis('off')

        ax.set_adjustable('box-forced')
def view_classify(img, ps, version="MNIST"):

    ''' Function for viewing an image and it's predicted classes.

    '''

    ps = ps.data.numpy().squeeze()



    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())

    ax1.axis('off')

    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    if version == "MNIST":

        ax2.set_yticklabels(np.arange(10))

    elif version == "Fashion":

        ax2.set_yticklabels(['T-shirt/top',

                            'Trouser',

                            'Pullover',

                            'Dress',

                            'Coat',

                            'Sandal',

                            'Shirt',

                            'Sneaker',

                            'Bag',

                            'Ankle Boot'], size='small');

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()
#Load and preprocess dataset for training and validation
data_dir='../input/flower_data/flower_data' 

train_dir = data_dir + '/train'

valid_dir = data_dir + '/valid'

test_dir='../input/test set'
# Define transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([

    transforms.RandomRotation(30),

    transforms.RandomResizedCrop(size=224),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



validation_transforms = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



# Load the datasets with ImageFolder

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

test_dataset = datasets.ImageFolder(test_dir, transform=validation_transforms)



# Using the image datasets and the transforms, define the dataloaders

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64, num_workers=4)

valid_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=64, num_workers=4)

test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=16, num_workers=4)
len(test_dataset)
#Visualizing few images

image, label = next(iter(train_dataloader))

imshow(image[10])



#Visualizing few images

image, label = next(iter(test_dataloader))

imshow(image[10])
#Label mapping 

train_class_names = train_dataset.classes

valid_class_names=validation_dataset.classes

import json

with open('../input/cat_to_name.json', 'r') as f: 

    cat_to_name = json.load(f)

cat_to_name['102']
category_map = sorted(cat_to_name.items(), key=lambda x: int(x[0]))



category_names = [cat[1] for cat in category_map]
# changing categories to their actual names 

for i in range(0,len(train_class_names)):

    train_class_names[i] = cat_to_name.get(train_class_names[i])

    

for i in range(0,len(valid_class_names)):

    valid_class_names[i] = cat_to_name.get(valid_class_names[i])
#Building and training the classifier

#Transfer learning::Rather than building a model and training from scratch,

#we can leverage a pretrained model, and adjust the classifier (the last part of it) as needed to fit our needs. This saves a huge amount of time and effort. We using vgg model.
vgg16 = models.vgg16(pretrained=True)
vgg16
#Training:: We want to train the final layers of the model. 

#The following functions will run forward and backward propogation with pytorch against the training set, and then test against the validation set.
# Use GPU if it's available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Freeze parameters so we don't backprop through them

for param in vgg16.parameters():

    param.requires_grad = False



classifier = nn.Sequential(OrderedDict([

                          ('fc1', nn.Linear(25088, 4096)),

                          ('relu', nn.ReLU()),

                          ('dropout', nn.Dropout(p=0.5)),

                          ('fc2', nn.Linear(4096, 102)),

                          ('output', nn.LogSoftmax(dim=1))

                          ]))

    

vgg16.classifier = classifier

vgg16.class_idx_mapping = train_dataset.class_to_idx
criterion = nn.NLLLoss()

optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.0001)
def validation(model, testloader, criterion, device):

    test_loss = 0

    accuracy = 0

    model.to(device)

    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)

        # images.resize_(images.shape[0], 3, 224, 224)



        output = model.forward(images)

        test_loss += criterion(output, labels).item()



        ps = torch.exp(output)

        equality = (labels.data == ps.max(dim=1)[1])

        accuracy += equality.type(torch.FloatTensor).mean()

    

    return test_loss, accuracy
def train(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cuda'): #cuda in kernel

    steps = 0

    

    # Change to train mode if not already

    model.train()

    # change to cuda

    model.to(device)



    for e in range(epochs):

        running_loss = 0



        for (images, labels) in trainloader:

            steps += 1



            images, labels = images.to(device), labels.to(device)



            optimizer.zero_grad()



            # Forward and backward passes

            outputs = model.forward(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()



            running_loss += loss.item()



            if steps % print_every == 0:

                

                # Make sure network is in eval mode for inference

                model.eval()



                # Turn off gradients for validation, saves memory and computations

                with torch.no_grad():

                    validation_loss, accuracy = validation(model, validloader, criterion, device)



                print("Epoch: {}/{}.. ".format(e+1, epochs),

                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),

                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validloader)),

                      "Validation Accuracy: {:.3f}".format((accuracy/len(validloader))*100))



                model.train()

                

                running_loss = 0
#Testing the network::

#train(model=vgg16, 

  ##      trainloader=train_dataloader, 

    ##    validloader=valid_dataloader,

      #  epochs=3, 

     #   print_every=20, 

      #  criterion=criterion,

       # optimizer=optimizer,

       # device="cuda") # cuda in kernel

#Testing the network::

train(model=vgg16, 

        trainloader=train_dataloader, 

        validloader=valid_dataloader,

        epochs=9, 

        print_every=40, 

        criterion=criterion,

        optimizer=optimizer,

        device="cuda") # cuda in kernel
#Testing on test set  can not be done as no test labels given.We can go for predicting class

def check_accuracy_on_test(testloader, model):    

    correct = 0

    total = 0

    model.to(device)

    with torch.no_grad():

        for data in testloader:

            images, labels = data

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            #print(labels)

            #print(outputs)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            print(total)

            #print(predicted)

            correct += (predicted == labels).sum().item()

            print(correct)

    #print(correct)



    return  correct / total
#Save the check point

vgg16.class_idx_mapping = train_dataset.class_to_idx

vgg16.class_idx_mapping
def save_checkpoint(state, filename='checkpoint.pth'):

    torch.save(state, filename)
save_checkpoint({

            'epoch': 9,

            'classifier': vgg16.classifier,

            'state_dict': vgg16.state_dict(),

            'optimizer' : optimizer.state_dict(),

            'class_idx_mapping': vgg16.class_idx_mapping,

            'arch': "vgg16"

            })
#Loading check point

def load_model(model_checkpoint):

    """

    Loads the model from a checkpoint file.



    Arguments: 

        model_checkpoint: Path to checkpoint file

    

    Returns: 

        model: Loaded model.

        idx_class_mapping: Index to class mapping for further evaluation.

    """



    checkpoint = torch.load(model_checkpoint)

    

    model = models.vgg16(pretrained=True)

    

    for param in model.parameters():

        param.requires_grad = False



    model.classifier = checkpoint["classifier"]

    model.load_state_dict(checkpoint["state_dict"])

    

    return model
model = load_model(model_checkpoint="checkpoint.pth")
#Class inference



img = Image.open("../input/flower_data/flower_data/train/61/image_06281.jpg")

print("Original image with size: {}".format(img.size))

plt.imshow(img)
def process_image(img_path):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,

        returns an Numpy array

    '''

    img = Image.open(img_path)

    w, h = img.size

    if w<h:

        size = 256, 999999999

    else:

        size = 999999999, 256



    img.thumbnail(size=size)

    

    w, h = img.size

    left = (w - 224) / 2

    right = (w + 224) / 2

    top = (h - 224) / 2

    bottom = (h + 224) / 2

    

    img = img.crop((left, top, right, bottom))

    

    # Convert to numpy array

    np_img = np.array(img)/255

    

    # Normalize

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    np_img = (np_img - mean) / std

    

    np_img = np_img.transpose(2, 0, 1)

    

    return np_img
img = process_image("../input/flower_data/flower_data/train/61/image_06281.jpg")

print("After resizing, cropping and normalizing, size: {}".format(img.shape))

#Creating Dataframe to get test images names and its corresponding lablel names
import pandas as pd
#Class prediction

class_idx_mapping = train_dataset.class_to_idx

idx_class_mapping = {v: k for k, v in class_idx_mapping.items()}

idx_class_mapping[1]
#Prediction Method

def predict(test_loader, model_checkpoint, topk=1, device="cpu", idx_class_mapping=idx_class_mapping):

    # Build the model from the checkpoint

    model = load_model(model_checkpoint)

    # No need for GPU

    model.to(device)

    

    model.eval()

    

    labels=[]

    

    with torch.no_grad():

        for images, _ in test_dataloader:

            images = images.to(device)

            output = model(images)

            ps = torch.exp(output)

            top_p, top_class = ps.topk(1, dim=1) 

            for i in top_class:

                #print(idx_class_mapping[i.item()])

                labels.append(idx_class_mapping[i.item()] )

                #print(i.item())

                      

               

    return labels
class_label=predict(test_dataloader,"checkpoint.pth", idx_class_mapping=idx_class_mapping)
print(len(class_label))
class_label[0]
#Methods to get list of Images names in test floder::

image_col=[]

for img_filename in os.listdir('../input/test set/test set'):

    image_col.append(img_filename)

print(len(image_col))
category_map = sorted(cat_to_name.items(), key=lambda x: int(x[0]))

plant_name=[]

for label in class_label:

    name=cat_to_name[label]

    #print(name)

    plant_name.append(name)

    #plant_name+=category_map[int(label)][1]

    #print(plant_name)

print(len(plant_name))

#print(category_map[int(class_label[3])][1] )
flower_dataframe = pd.DataFrame({'image_name': image_col, 'class_label': class_label,'plant_name': plant_name})

flower_dataframe.sort_values('image_name')