# Imports here

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import json
data_dir = '../input/flowers/flower_data/flower_data/'

train_dir = data_dir + 'train'
valid_dir = data_dir + 'valid'

import os
print(os.listdir("../input/flowers/flower_data/flower_data"))
def preprocess_data(train_dir, valid_dir, 
                    norm_mean=[0.485, 0.456, 0.406], norm_stdv=[0.229, 0.224, 0.225]):
    '''
    Inputs:
    train_dir, valid_dir, test_dir - data directories
    norm_mean, norm_stdv - image mean and standard deviation for 3 channels
    
    Returns:
    trainloader, validloader, testloader - torchvision data loaders
    '''
    data_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(norm_mean, norm_stdv)])

    data_transforms_train = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(p=0.1),
                                           transforms.ToTensor(),
                                           transforms.Normalize(norm_mean, norm_stdv)])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    
    # Save class ids
    class_idx = train_data.class_to_idx
    return trainloader, validloader, class_idx
# TODO: Define your transforms for the training, validation, and testing sets
norm_mean = [0.485, 0.456, 0.406]
norm_stdv = [0.229, 0.224, 0.225]

trainloader, validloader, class_idx = preprocess_data(train_dir, valid_dir, norm_mean, norm_stdv)
with open('../input/labels1/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# select and load pretrained model
def load_pretrained_model(arch = 'vgg'):

    if arch == 'vgg':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = 1024
        
    #Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model, input_size

arch = 'vgg'
arch = 'densenet'
model, input_size = load_pretrained_model(arch)
print(model, input_size)
def define_feedforward_classification_for_model(model, input_size, output_size, prob_dropout):
    
    hidden_sizes = [1024, 1024]
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=prob_dropout)),
                          ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p=prob_dropout)),
                          ('output', nn.Linear(hidden_sizes[1], output_size)),
                          ('logsoftmax', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier

output_size=102
define_feedforward_classification_for_model(model, input_size, output_size=output_size, prob_dropout=0.5)
print(model)
def train_classifier(model, trainloader, validloader, criterion, optimizer, epochs, gpu):
    """
    Trains the selected model. Performs validation loop every 40 steps and prints progress. 
    Inputs:
        model - CNN architecture to be trained
        trainloader - pytorch data loader of training data
        validloader - pytorch data loader of data to be used to validate.
        criterion - loss function to be executed (default- nn.NLLLoss)
        optimizer - optimizer function to apply gradients (default- adam optimizer)
        epochs - number of epochs to train on
        gpu - boolean that flags GPU use
    Returns:
        model - Trained CNN
    """
    steps = 0
    print_every = 40
    run_loss = 0
    
    #Selects CUDA processing if gpu == True and if the environment supports CUDA
    if gpu and torch.cuda.is_available():
        print('GPU TRAINING')
        model.cuda()
    elif gpu and torch.cuda.is_available() == False:
        print('GPU processing selected but no NVIDIA drivers found... Training under CPU.')
    else:
        print('CPU TESTING')
        
    for e in range(epochs):
              
        model.train()
        
        #Training forward pass and backpropagation
        for images, labels in iter(trainloader):
            steps+= 1
            images, labels = Variable(images), Variable(labels)
            if gpu and torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()

            out = model.forward(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.data.item()
            
            #Runs validation forward pass and loop at specified interval
            if steps % print_every == 0:
                model.eval()

                acc = 0
                valid_loss = 0

                for images, labels in iter(validloader):
                    images, labels = Variable(images), Variable(labels)
                    if gpu and torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()
                    with torch.no_grad():
                        out = model.forward(images)
                        valid_loss += criterion(out, labels).data.item()
        
                        ps = torch.exp(out).data
                        equality = (labels.data == ps.max(1)[1])
        
                        acc += equality.type_as(torch.FloatTensor()).mean()
        
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                 "Training Loss: {:.3f}.. ".format(run_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}".format(acc/len(validloader)))  
    
                run_loss = 0
                model.train()
            
    print('{} EPOCHS COMPLETE. MODEL TRAINED.'.format(epochs))
    return model
learning_rate = 0.001
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
epochs=15
gpu=True
model = train_classifier(model, trainloader, validloader, criterion=criterion, optimizer=optimizer, epochs=epochs, gpu=gpu)
# TODO: Save the checkpoint 

def save_model_checkpoint(model, input_size, epochs, save_dir, arch, learning_rate, class_idx, optimizer, criterion, output_size):
    """
    Save trained model as checkpoint file.
    Parameters:
        model - Previously trained and tested CNN
        input_size - Input size used on the specific CNN
        epochs - Number of epochs used to train the CNN
        save_dir - Directory to save the checkpoint file(default- current path)
        arch - pass string value of architecture used for loading
    Returns:
        None
    """
    saved_model = {
    'input_size':input_size,
    'epochs':epochs,
    'arch':arch,
    'hidden_units':[each.out_features for each in model.classifier if hasattr(each, 'out_features') == True],
    'output_size': output_size,
    'learning_rate': learning_rate,
    'class_to_idx': class_idx,
    'optimizer_dict': optimizer.state_dict(),
    'criterion_dict': criterion.state_dict(),
    'classifier': model.classifier,
    'state_dict': model.state_dict() 
    }
    #Save checkpoint in current directory unless otherwise specified by save_dir
    if len(save_dir) == 0:
        save_path = save_dir + 'checkpoint.pth'
    else:
        save_path = save_dir + '/checkpoint.pth'
    torch.save(saved_model, save_path)
    print('Model saved at {}'.format(save_path))
# create save_dir for checkpoints
import os

save_dir = './save_dir'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_model_checkpoint(model, input_size=input_size, epochs=epochs, save_dir=save_dir, arch=arch, learning_rate=learning_rate, class_idx=class_idx, optimizer=optimizer, criterion=criterion, output_size=output_size)

print(os.listdir("./save_dir"))
# TODO: Save the checkpoint 


# TODO: Write a function that loads a checkpoint and rebuilds the model


#def process_image(image):
 #   ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        #returns an Numpy array
  #  '''
    
    # TODO: Process a PIL image for use in a PyTorch model


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
#def predict(image_path, model, topk=5):
 #   ''' Predict the class (or classes) of an image using a trained deep learning model.
  #  '''
    
    # TODO: Implement the code to predict the class from an image file

def predict(image_path, model, top_num=5):
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
# TODO: Display an image along with the top 5 classes

def plot_solution(image_path, model):
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)

    # Set up title
    flower_num = image_path.split('/')[1]
    title_ = label_map[flower_num]

    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_);

    # Make prediction
    probs, labs, flowers = predict(image_path, model) 

    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()