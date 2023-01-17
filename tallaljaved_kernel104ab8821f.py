# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch import nn, optim

from torchvision import transforms, models, datasets

import json

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

train_data_path = "../input/flower_data/flower_data/train"

validation_data_path = "../input/flower_data/flower_data/valid"

test_data_path = "../input/test set/test set"



# Any results you write to the current directory are saved as output.

cat_dic = json.load(open("../input/cat_to_name.json"))

print(cat_dic["21"])
#load the data using a generator

train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],

                                                            [0.229, 0.224, 0.225])])



validation_transforms = transforms.Compose([transforms.Resize(255),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],

                                                           [0.229, 0.224, 0.225])])



# Pass transforms in here, then run the next cell to see how the transforms look

train_data = datasets.ImageFolder(train_data_path , transform=train_transforms)

validation_data = datasets.ImageFolder(validation_data_path, transform=validation_transforms)

#test_data = datasets.ImageFolder(test_data_path, transforms = validation_transforms)



trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)

#testlodaerr = torch.utils.data.DataLoader(test_data, batch_size=64)
# checking the distribution of data

def count_samples_in_class(images, classes):

    samples = [0] * len(classes)

    for item in images:

        samples[item[1]] +=1

    return samples

    

samples = count_samples_in_class(train_data.imgs, train_data.classes)

samples = torch.FloatTensor(samples)



categories = np.array(list(cat_dic.values()), dtype=str)

d = pd.DataFrame(categories, columns=['Name'])

d['Frequency'] = samples

print(f'Total Images:{len(train_data.imgs)}')

print(f'mean of dataset is:{torch.mean(samples)}')

print(f'STd of dataset is: {torch.std(samples)}')

print(f'Variance of dataset is: {torch.var(samples)}')

plt.boxplot(samples)

plt.show()

#plt.plot(samples)

plt.plot(samples.numpy())

#plt.plot(samples.numpy())

d.T

# util for sampling balanced data - and checking the class distribution

sampler = torch.utils.data.sampler.RandomSampler(train_data, replacement=True, num_samples=64)                     

                                                                                

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,                               

                                                             sampler = sampler)  

#trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

#print(weights)

#download the model

model = models.resnext101_32x8d(pretrained=True,progress=True)

#remove the final layer from model

orig_model = model



# freezing with no backprop

for param in model.parameters():

    param.requires_grad = False



from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([

                          ('fc1', nn.Linear(2048, 1000)),

                          ('relu', nn.ReLU()),

                          ('fc2', nn.Linear(1000, 102)),

                          ('output', nn.LogSoftmax(dim=1))

                          ]))

    

model.fc = classifier

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen

optimizer = optim.SGD(model.fc.parameters(), lr=0.1)



#print(model)
import time

# Use GPU if it's available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# shift to device the mdoel

model.to(device);

print(device)



epochs = 45

steps = 0

running_loss = 0

print_every = 5

for epoch in range(epochs):

    

    # for the logs in kaggle kernel.

    print_string = f"echo epoch is: {epoch}"

    os.system(print_string)

    

    # after 35th epoch slow down the learning rate as per rexnet paper

    if epoch == 30:

        for g in optimizer.param_groups:

            g['lr'] = 0.001

    

    for inputs, labels in trainloader:

        steps += 1

        # Move input and label tensors to the default device

        inputs, labels = inputs.to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        logps = model.forward(inputs)

        loss = criterion(logps, labels)

        loss.backward()

        optimizer.step()



        running_loss += loss.item()

        

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0

            model.eval()

            with torch.no_grad():

                for inputs, labels in validationloader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)

                    batch_loss = criterion(logps, labels)

                    

                    test_loss += batch_loss.item()

                    

                    # Calculate accuracy

                    ps = torch.exp(logps)

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    

            print(f"Epoch {epoch+1}/{epochs}.. "

                  f"Step: {steps}..."

                  f"Train loss: {running_loss/print_every:.3f}.. "

                  f"Test loss: {test_loss/len(validationloader):.3f}.. "

                  f"Test accuracy: {accuracy/len(validationloader):.3f}")

            running_loss = 0

            model.train()
# Process our image

def process_image(image_path):

    # Load Image

    img = Image.open(image_path)

    

    # Get the dimensions of the image

    width, height = img.size

    

    # Resize by keeping the aspect ratio, but changing the dimension

    # so the shortest size is 255px

    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))

    

    # Get the dimensions of the new image size

    width, height = img.size

    

    # Set the coordinates to do a center crop of 224 x 224

    left = (width - 224)/2

    top = (height - 224)/2

    right = (width + 224)/2

    bottom = (height + 224)/2

    img = img.crop((left, top, right, bottom))

    

    # Turn image into numpy array

    img = np.array(img)

    

    # Make the color channel dimension first instead of last

    img = img.transpose((2, 0, 1))

    

    # Make all values between 0 and 1

    img = img/255

    

    # Normalize based on the preset mean and standard deviation

    img[0] = (img[0] - 0.485)/0.229

    img[1] = (img[1] - 0.456)/0.224

    img[2] = (img[2] - 0.406)/0.225

    

    # Add a fourth dimension to the beginning to indicate batch size

    img = img[np.newaxis,:]

    

    # Turn into a torch tensor

    image = torch.from_numpy(img)

    image = image.float()

    return image
#Kindly run this for predictions. There might be some seems to be some issue with some of the images with respect to dimensions. So running a  :)

from PIL import *

files = os.listdir("../input/test set/test set")

index = 0

print(len(files))

for file in files:

    image = process_image(test_data_path+"/"+file)

    image = image.to(device)

    try:

        output = model.forward(image)

        # Reverse the log function in our output

        output = torch.exp(output)



        # Get the top predicted class, and the output percentage for that class

        probs, classes = output.topk(1, dim=1)

        class_name = cat_dic[str(classes.item())]

        print (f"{index}: "

                f"file name: {file},  "

                f"Predicted Class: {class_name},  "

                f"Class Number: {classes.item()}")

        index+=1

    except:

        print("An exception occurred")

        continue

        index+=1

    