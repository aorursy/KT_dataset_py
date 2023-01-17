%ls "../input/notebook-ims/"
import os

import numpy as np

import torch



import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt



%matplotlib inline
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
# define training and test data directories

data_dir = 'flower_photos/'

train_dir = os.path.join(data_dir, 'train/')

test_dir = os.path.join(data_dir, 'test/')



# classes are folders in each directory with these names

classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
# load and transform data using ImageFolder



# VGG-16 Takes 224x224 images as input, so we resize all of them

data_transform = transforms.Compose([transforms.RandomResizedCrop(224), 

                                      transforms.ToTensor()])



train_data = datasets.ImageFolder(train_dir, transform=data_transform)

test_data = datasets.ImageFolder(test_dir, transform=data_transform)



# print out some data stats

print('Num training images: ', len(train_data))

print('Num test images: ', len(test_data))
# define dataloader parameters

batch_size = 20

num_workers=0



# prepare data loaders

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 

                                           num_workers=num_workers, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 

                                          num_workers=num_workers, shuffle=True)
# Visualize some sample data



# obtain one batch of training images

dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))

    ax.set_title(classes[labels[idx]])
# Load the pretrained model from pytorch

vgg16 = models.vgg16(pretrained=True)



# print out the model structure

print(vgg16)
print(vgg16.classifier[6].in_features) 

print(vgg16.classifier[6].out_features) 

# Freeze training for all "features" layers

for param in vgg16.features.parameters():

    param.requires_grad = False

    
import torch.nn as nn## TODO: add a last linear layer  that maps n_inputs -> 5 flower classes

## new layers automatically have requires_grad = True









# after completing your model, if GPU is available, move the model to GPU

if train_on_gpu:

    vgg16.cuda()
import torch.optim as optim



# specify loss function (categorical cross-entropy)

criterion = nn.CrossEntropyLoss()



# specify optimizer (stochastic gradient descent) and learning rate = 0.001

optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)
# number of epochs to train the model

n_epochs = 2



## TODO complete epoch and training batch loops

## These loops should update the classifier-weights of this model

## And track (and print out) the training loss over time





# track test loss 

# over 5 flower classes

test_loss = 0.0

class_correct = list(0. for i in range(5))

class_total = list(0. for i in range(5))



vgg16.eval() # eval mode



# iterate over test data

for data, target in test_loader:

    # move tensors to GPU if CUDA is available

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = vgg16(data)

    # calculate the batch loss

    loss = criterion(output, target)

    # update  test loss 

    test_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)    

    # compare predictions to true label

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class

    for i in range(batch_size):

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



# calculate avg test loss

test_loss = test_loss/len(test_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(5):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
# obtain one batch of test images

dataiter = iter(test_loader)

images, labels = dataiter.next()

images.numpy()



# move model inputs to cuda, if GPU available

if train_on_gpu:

    images = images.cuda()



# get sample outputs

output = vgg16(images)

# convert output probabilities to predicted class

_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())



# plot the images in the batch, along with predicted and true labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))

    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),

                 color=("green" if preds[idx]==labels[idx].item() else "red"))