# Imports here
import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.__version__

import os
print(os.listdir("../input/"))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from PIL import Image

data_dir = '../input/flower_data/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
"""data_dir_1 = str(np.random.randint(1,103))
print("Class Directory: ",data_dir_1)
for file_name in os.listdir(os.path.join(train_dir, data_dir_1))[1:3]:
    img_array = cv2.imread(os.path.join(train_dir, data_dir_1, file_name))
    img_array = cv2.resize(img_array,(224, 224), interpolation = cv2.INTER_CUBIC)
    plt.imshow(img_array)
    plt.show()
    print(img_array.shape)"""
classes = os.listdir(valid_dir)
# TODO: Define your transforms for the training and validation sets
data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# TODO: Load the datasets with ImageFolder
train_data_set = datasets.ImageFolder(train_dir, transform=data_transforms)
test_data_set = datasets.ImageFolder(valid_dir, transform=data_transforms)

#print('Num training images: ' + str(len(train_data_set)))
#print('Num test images: ' + str(len(test_data_set)))

# TODO: Using the image datasets and the trainforms, define the dataloaders
batch_size = 32
num_workers = 0

train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)

"""dataiter = iter(test_loader)

images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])"""
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
import json

with open('../input/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# TODO: Build and train your network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True)
#print(model)
#print(model.classifier[6].in_features)

for param in model.features.parameters():
    param.requires_grad = False
    
# Freeze training for all "features" layers
#for _, param in model.named_parameters():
#    param.requires_grad = False
n_inputs = model.classifier[6].in_features
from collections import OrderedDict

model.fc = nn.Linear(n_inputs, len(classes)) #nn.Linear(n_inputs, len(classes))
if train_on_gpu:
    model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)
# Define epochs (between 50-200)
epochs = 80
# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity

# Some lists to keep track of loss and accuracy during each epoch
epoch_list = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
# Start epochs
for epoch in range(epochs):
    
    #adjust_learning_rate(optimizer, epoch)
    #exp_lr_scheduler.step()
    # monitor training loss
    train_loss = 0.0
    val_loss = 0.0
    
    ###################
    # train the model #
    ###################
    # Set the training mode ON -> Activate Dropout Layers
    model.train() # prepare model for training
    # Calculate Accuracy         
    correct = 0
    total = 0
    
    # Load Train Images with Labels(Targets)
    for data, target in train_loader:
        
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        
        if type(output) == tuple:
            output, _ = output
        
        # Calculate Training Accuracy 
        predicted = torch.max(output.data, 1)[1]        
        # Total number of labels
        total += len(target)
        # Total correct predictions
        correct += (predicted == target).sum()
        
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
    
    # calculate average training loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)
    
    # Avg Accuracy
    accuracy = 100 * correct / float(total)
    
    # Put them in their list
    train_acc_list.append(accuracy)
    train_loss_list.append(train_loss)
    
        
    # Implement Validation like K-fold Cross-validation 
    
    # Set Evaluation Mode ON -> Turn Off Dropout
    model.eval() # Required for Evaluation/Test

    # Calculate Test/Validation Accuracy         
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:


            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Predict Output
            output = model(data)
            if type(output) == tuple:
                output, _ = output

            # Calculate Loss
            loss = criterion(output, target)
            val_loss += loss.item()*data.size(0)
            # Get predictions from the maximum value
            predicted = torch.max(output.data, 1)[1]

            # Total number of labels
            total += len(target)

            # Total correct predictions
            correct += (predicted == target).sum()
    
    # calculate average training loss and accuracy over an epoch
    val_loss = val_loss/len(test_loader.dataset)
    accuracy = 100 * correct/ float(total)
    
    # Put them in their list
    val_acc_list.append(accuracy)
    val_loss_list.append(val_loss)
    
    # Print the Epoch and Training Loss Details with Validation Accuracy   
    print('Epoch: {} \tTraining Loss: {:.4f}\t Val. acc: {:.2f}%'.format(
        epoch+1, 
        train_loss,
        accuracy
        ))
    # save model if validation loss has decreased
    if val_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        val_loss))
        # Save Model State on Checkpoint
        torch.save(model.state_dict(), 'model_1.pt')
        valid_loss_min = val_loss
    # Move to next epoch
    epoch_list.append(epoch + 1)
model.class_to_idx = train_data_set.class_to_idx

checkpoint = {
    'arch': 'vgg16',
    'class_to_idx': model.class_to_idx, 
    'state_dict': model.state_dict(),
    'input_size': 4096,
    'output_size': 102
}

torch.save(checkpoint, 'vgg16-model.pth')
val_acc = sum(val_acc_list[:]).item()/len(val_acc_list)
print("Validation Accuracy of model = {} %".format(val_acc))
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
img = images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

model.eval() # Required for Evaluation/Test
# get sample outputs
output = model(images)
if type(output) == tuple:
            output, _ = output
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(20, 5))
for idx in np.arange(12):
    ax = fig.add_subplot(3, 4, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(img[idx], (1, 2, 0)))
    ax.set_title("Pr: {} Ac: {}".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
# TODO: Write a function that loads a checkpoint and rebuilds the model

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

state_dict = load_checkpoint('vgg16-model.pth')
#print(state_dict.keys())
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
%matplotlib inline

_= imshow(process_image('../input/flower_data/flower_data/train/1/image_06734.jpg'))
# Implement the code to predict the class from an image file
def predict(image, checkpoint, filepath, topk=5, labels='', gpu=False):
    '''Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use command line values when specified
    """if args.image:
        image = args.image     
        
    if args.checkpoint:
        checkpoint = args.checkpoint

    if args.topk:
        topk = args.topk
            
    if args.labels:
        labels = args.labels

    if args.gpu:
        gpu = args.gpu
    """
    
    # Load the checkpoint
    checkpoint_dict = torch.load(checkpoint)

    num_labels = len(checkpoint_dict['class_to_idx'])
    
    model = load_checkpoint(checkpoint)
    
    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        model.cuda()
        
    training = model.training    
    model.eval()
    
    image = process_image(image)
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
    
    if gpu and torch.cuda.is_available():
        image = image.cuda()
            
    result = model(image).topk(topk)

    if gpu and torch.cuda.is_available():
        probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
        classes = result[1].data.cpu().numpy()[0]
    else:       
        probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
        classes = result[1].data.numpy()[0]

    labels = list(cat_to_name.values())
    classes_aux = []
    for c in classes:
        classes_aux.append(labels[c])
        
    model.train(mode = training)

    #if args.image:
    #    print('Predictions & probabilities:', list(zip(classes, probs)))
    
    return probs, classes_aux
image_path = '../input/flower_data/flower_data/train/3/image_06612.jpg'
flower_image = mpimg.imread(image_path)

f,ax_array = plt.subplots(2,1)
ax_array[0].imshow(flower_image)
ax_array[0].set_title('Fire Lily')

probabilities, classes = predict(image=image_path,checkpoint='vgg16-model.pth',filepath='',gpu=True)
y_position = np.arange(len(classes))
ax_array[1].barh(y_position, probabilities,align='center',alpha=0.5)
ax_array[1].set_yticks(y_position)
ax_array[1].set_yticklabels(classes)
ax_array[1].invert_yaxis()
_ =ax_array[1].set_xlabel('Probabilities')