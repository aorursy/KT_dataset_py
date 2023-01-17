import numpy as np
import torch    
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
train_dir = r'../input/waste-classification-data/DATASET/TRAIN'
test_dir = r'../input/waste-classification-data/DATASET/TEST'

classes = ['O', 'R']
#according to pytorch documention, for pretrained models all images have to have width and height of at least 224 pixels
#and have to be normalized

#let's load and transform data

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize])

train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

#now check if they've loaded correctly
print("Number of train images: ", (len(train_data)))
print("Number of test images: ", len(test_data))
#prepare data loaders
batch_size = 20

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
#let's visualize one batch of training data

dataiter = iter(train_loader)
images, labels = dataiter.next()
images.numpy() #convert images to numpy

#plot images with corresponding labels

fig = plt.figure(figsize=(25, 4))
for image in range(batch_size):
    ax = fig.add_subplot(2, 20/2, image+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[image], (1, 2, 0)))
    ax.set_title(classes[labels[image]])
# download pretrained model, vgg16

vgg16 = models.vgg16(pretrained=True)

# check model's architecture
print(vgg16)
# now we need to freeze the model's parameters, so it acts as a fixed feature extractor and then
# we'll replace last linear layer so that it'll have only 2 out_features -> the number of our classes

for param in vgg16.features.parameters():
    param.requires_grad = False

# access the last layer in the net
n_inputs = vgg16.classifier[6].in_features

# create new layer to have original number of in_features and out_features equal to number of classes we have in our data
last_layer = nn.Linear(n_inputs, len(classes))

# overwrite last layer with our layer
vgg16.classifier[6] = last_layer

# check if it's correct
print(vgg16)
#final thing before training is to specify loss function, optimzier and learning rate

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer and learning rate = 0.001
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)
# if GPU is available, move the model to GPU
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    vgg16.cuda()
#train the network

n_epochs = 1

for epoch in range(1, n_epochs + 1):
    
    #vgg16 by default is in train() mode so we don't have to use this method here
    
    #keep track of training loss
    train_loss = 0
    
    for batch_i, (data, target) in enumerate(train_loader):
        
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
           
        optimizer.zero_grad()             #clear the gradients
        output = vgg16(data)              #do the forward pass
        loss = criterion(output, target)  #calculate loss
        loss.backward()                   #do the backward pass
        optimizer.step()                  #perform parameter update
        train_loss += loss.item()         #update training loss
        
        if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
            print('Epoch %d, Batch %d loss: %.16f' %
                  (epoch, batch_i + 1, train_loss / 20))
            train_loss = 0.0
#test the network

test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

vgg16.eval() # now we need to switch to eval mode

for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    
    output = vgg16(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    
    
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)  
    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    
    # calculate test accuracy for each object class
    try:
        for i in range(20):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    except:
        for i in range(13):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

# calculate avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(len(classes)):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
#visualize results

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
for image in range(batch_size):
    ax = fig.add_subplot(2, 20/2, image+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[image].cpu(), (1, 2, 0)))
    ax.set_title("{} ({})".format(classes[preds[image]], classes[labels[image]]),
                 color=("green" if preds[image]==labels[image].item() else "red"))