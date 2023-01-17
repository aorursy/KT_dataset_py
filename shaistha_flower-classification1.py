# Import resources

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import time

import json

import copy



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

import PIL

import pytesseract

from PIL import Image

from collections import OrderedDict





import torch

from torch import nn, optim

from torch.optim import lr_scheduler

from torch.autograd import Variable

import torchvision

from torchvision import datasets, models, transforms

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn

import torch.nn.functional as F



import os
# check if GPU is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('NO GPU Found..! Training on CPU ...')

else:

    print('You are good to go!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#load the data 

data_dir = '../input/flower_data/flower_data'

train_dir = data_dir + '/train'

valid_dir = data_dir + '/valid'

test_dir = '../input/test set'
# Define your transforms for the training and testing sets

data_transforms = {

    'train': transforms.Compose([

        transforms.RandomRotation(30),

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ]),

    'valid': transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ]),

    'test set': transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ])

}

# Load the datasets with ImageFolder

dirs = {'train': train_dir, 

        'valid': valid_dir, 

        'test set' : test_dir}

image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test set']}

#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),

                                        #  data_transforms[x])

                #  for x in ['train', 'valid']}



# Using the image datasets and the trainforms, define the dataloaders



dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=20, shuffle=True) for x in ['train', 'valid', 'test set']}

#batch_size = 20

#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,

                                           #  shuffle=True, num_workers=4)

             # for x in ['train', 'valid']}



class_names = image_datasets['train'].classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test set']}

class_names = image_datasets['train'].classes
print(dataset_sizes)

print(device)
# Label mapping

with open('../input/cat_to_name.json', 'r') as f:

    cat_to_name = json.load(f)
#build and train classifier

# Run this to test the data loader

images, labels = next(iter(dataloaders['train']))

images.size()
 # Run this to test your data loader

images, labels = next(iter(dataloaders['train']))

rand_idx = np.random.randint(len(images))

# print(rand_idx)

print("label: {}, class: {}, name: {}".format(labels[rand_idx].item(),

                                               class_names[labels[rand_idx].item()],

                                               cat_to_name[class_names[labels[rand_idx].item()]]))
model_name = 'densenet' #vgg

if model_name == 'densenet':

    model = models.densenet161(pretrained=True)

    num_in_features = 2208

    print(model)

elif model_name == 'vgg':

    model = models.vgg19(pretrained=True)

    num_in_features = 25088

    print(model.classifier)

else:

    print("Unknown model, please choose 'densenet' or 'vgg'")
# Create classifier

for param in model.parameters():

    #we don’t update the weights from the pre-trained model.

    param.requires_grad = False



def build_classifier(num_in_features, hidden_layers, num_out_features):

   

    classifier = nn.Sequential()

    if hidden_layers == None:

        classifier.add_module('fc0', nn.Linear(num_in_features, 102))

    else:

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

        classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))

        classifier.add_module('relu0', nn.ReLU())

        classifier.add_module('drop0', nn.Dropout(.6))

        classifier.add_module('relu1', nn.ReLU())

        classifier.add_module('drop1', nn.Dropout(.5))

        for i, (h1, h2) in enumerate(layer_sizes):

            classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))

            classifier.add_module('relu'+str(i+1), nn.ReLU())

            classifier.add_module('drop'+str(i+1), nn.Dropout(.5))

        classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))

        

    return classifier
hidden_layers = None 



classifier = build_classifier(num_in_features, hidden_layers, 102)

print(classifier)



 # Only train the classifier parameters, feature parameters are frozen

if model_name == 'densenet':

    model.classifier = classifier

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adadelta(model.parameters()) # Adadelta #weight optim.Adam(model.parameters(), lr=0.001, momentum=0.9)

    #optimizer_conv = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.9)

    sched = optim.lr_scheduler.StepLR(optimizer, step_size=4)

elif model_name == 'vgg':

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

    sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

else:

    pass
def train_model(model, criterion, optimizer, sched, num_epochs=5):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:

            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        #sched.step()

                        loss.backward()

                        

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'valid' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    #load best model weights

    model.load_state_dict(best_model_wts)

    

    return model
epochs = 10

model.to(device)

model = train_model(model, criterion, optimizer, sched, epochs)



# Evaluation



model.eval()



accuracy = 0



for inputs, labels in dataloaders['valid']:

    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs)

    

    # Class with the highest probability is our predicted class

    equality = (labels.data == outputs.max(1)[1])



    # Accuracy is number of correct predictions divided by all predictions

    accuracy += equality.type_as(torch.FloatTensor()).mean()

    

print("Validation accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))
# Saving the checkpoint

model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'input_size': 2208,

              'output_size': 102,

              'epochs': epochs,

              'batch_size': 64,

              'model': models.densenet161(pretrained=True),

              'classifier': classifier,

              'scheduler': sched,

              'optimizer': optimizer.state_dict(),

              'state_dict': model.state_dict(),

              'class_to_idx': model.class_to_idx

             }

   

torch.save(checkpoint, 'checkpoint_ic_d161.pth')
# Loading the checkpoint

ckpt = torch.load('checkpoint_ic_d161.pth')

ckpt.keys()
# Load a checkpoint and rebuild the model

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    model = checkpoint['model']

    model.classifier = checkpoint['classifier']

    model.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']

    optimizer = checkpoint['optimizer']

    epochs = checkpoint['epochs']

    

    for param in model.parameters():

        param.requires_grad = False

        

    return model, checkpoint['class_to_idx']
model, class_to_idx = load_checkpoint('checkpoint_ic_d161.pth')

model
idx_to_class = { v : k for k,v in class_to_idx.items()}
#Inference for Classification

#Image Preprocessing

image_path = '../input/test set/test set/ab45.jpg'

img = Image.open(image_path)
def process_image(image):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,

        returns an Numpy array

    '''

    # Process a PIL image for use in a PyTorch model

    # tensor.numpy().transpose(1, 2, 0)

    preprocess = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], 

                             std=[0.229, 0.224, 0.225])

    ])

    image = preprocess(image)

    return image
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
with Image.open('../input/test set/test set/ab45.jpg') as image:

    plt.imshow(image)
#Class Prediction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.class_to_idx = image_datasets['train'].class_to_idx
def predict2(image_path, model, topk=5):

    ''' Predict the class (or classes) of an image using a trained deep learning model.

    '''

    

    # Implement the code to predict the class from an image file

    img = Image.open(image_path)

    img = process_image(img)

    

    # Convert 2D image to 1D vector

    img = np.expand_dims(img, 0)

    

    

    img = torch.from_numpy(img)

    

    model.eval()

    inputs = Variable(img).to(device)

    logits = model.forward(inputs)

    

    ps = F.softmax(logits,dim=1)

    topk = ps.cpu().topk(topk)

    

    return (e.data.numpy().squeeze().tolist() for e in topk)
#checking for one image

img_path = '../input/test set/test set/ab45.jpg'

probs, classes = predict2(img_path, model.to(device))

print(probs)

print(classes)

flower_names = [cat_to_name[class_names[e]] for e in classes]

print(flower_names)

def view_classify(img_path, prob, classes, mapping):

    ''' Function for viewing an image and it's predicted classes.

    '''

    image = Image.open(img_path)



    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)

    ax1.imshow(image)  

    flower_names = [cat_to_name[class_names[e]] for e in classes]

    flower_name =  flower_names[0]

    ax1.set_title(flower_name)

    ax1.axis('off')

    

    y_pos = np.arange(len(prob))

    ax2.barh(y_pos, prob, align='center')

    ax2.set_yticks(y_pos)

    ax2.invert_yaxis() 

    ax2.set_yticklabels(flower_names)

    # labels read top-to-bottom

    ax2.set_title('Class Probability')



view_classify(img_path, probs, classes, cat_to_name)
#saving the model for test set

torch.save(model.state_dict(), 'model1.pth')
#Prediction of all images in the test set



test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test set'])

dataloader = torch.utils.data.DataLoader(test_data)
data_labels = []

model.to(device)

model.eval()

with torch.no_grad():

    for images, labels in dataloader:

        images = images.to(device)

        ps = model.forward(images)

        

        if type(ps) == tuple:

            ps, _ = ps

        

        _, preds_tensor = torch.max(ps, 1)

        preds = np.squeeze(preds_tensor.numpy())if not device else np.squeeze(preds_tensor.cpu().numpy())

        data_labels.append(int(preds))
#load JSON file

file_path = "../input/cat_to_name.json"

df = pd.read_json(file_path, typ='series')

df = df.to_frame('category')

df.head()
df.count()
files =[]

categories = []

for file in os.listdir(os.path.join(test_dir, "test set")):

    files.append(file)



for i in data_labels:

    categories.append(df.loc[i+1, 'category'])    
#Converting Test Predictions to dataframe

d = {'Image_Name': files, 'Class_Label': data_labels, 'Flower_Category': categories}

result = pd.DataFrame(d)
result = result.sort_values(by="Image_Name")
result
result.to_csv("../working/result.csv")