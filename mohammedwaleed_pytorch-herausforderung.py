import torch
from torchvision import transforms, models, datasets
import torchvision
from torch.autograd import Variable
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import time
import matplotlib.pyplot as plt
plt.ion() 
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import os
import numpy as np
from PIL import Image
import copy
import cv2
import math
from PIL import Image


import helper

from collections import OrderedDict

train_on_gpu = torch.cuda.is_available()
def resize_image(input_image_path,
                 size):
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(size)
    resized_image.show()

data_dir = '../input/flowers-datasets/flowers_dataset/flower'


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
# number of subprocesses to use for data loading
#num_workers = 0
# how many samples per batch to load
batch_size = 16
# percentage of training set to use as validation
#test_size = 0.5


# TODO: Define your transforms for the training and validation sets
train_data_transforms = transforms.Compose([
                                       transforms.RandomRotation(10),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



# TODO: Load the datasets with ImageFolder
train_image_datasets = datasets.ImageFolder(train_dir,transform=train_data_transforms)
valid_image_datasets = datasets.ImageFolder(valid_dir,transform=valid_data_transforms)

#print(valid_image_datasets.class_to_idx)
# split the data into training - test set
#n_train_image = len(train_image_datasets)
#indices = list(range(n_train_image))
#np.random.shuffle(indices)
#split = int(np.floor(test_size * n_train_image))
#train_idx, test_idx = indices[split:], indices[:split]

#train_sampler = SubsetRandomSampler(train_idx)
#test_sampler = SubsetRandomSampler(test_idx)
# TODO: Using the image datasets and the trainforms, define the dataloaders
traindataloaders = torch.utils.data.DataLoader(train_image_datasets,batch_size=batch_size, shuffle=True)
validdataloaders = torch.utils.data.DataLoader(valid_image_datasets,batch_size=batch_size)
#testdataloaders  = torch.utils.data.DataLoader(train_image_datasets,batch_size=batch_size,sampler=test_sampler)

class_names = train_image_datasets.classes
def random_mini_batches(X, Y, mini_batch_size =16, seed = 0):

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[:,permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


import json

with open('../input/pytorch-challange-flower-dataset/cat_to_name.json', 'r') as f:
    
    cat_to_name = json.load(f)
classes_name = []

for i in range(len(cat_to_name)):
    classes_name.append(cat_to_name[str(i+1)])
# obtain one batch of training images
dataiter = iter(traindataloaders)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display


# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(16):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(cat_to_name[str(labels[idx].item() + 1)])
resnet_model = models.resnet152(pretrained=True)
deep_model = models.vgg19_bn(pretrained=True)
#deep_model = models.resnet152(pretrained=True)
deep_model
# Freeze training for all layers
for param in resnet_model.parameters():
    param.require_grad = False
    
for param in deep_model.parameters():
    param.require_grad = False

classifier = nn.Sequential(nn.Linear(25088,4096),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(4096,4096),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Linear(4096,102))
# Newly created modules have require_grad=True by default
num_features = resnet_model.fc.in_features
features = list()
features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
resnet_model.fc = nn.Sequential(*features) # Replace the model classifier
deep_model.classifier = classifier # Replace the model classifier

deep_model
resnet_model.load_state_dict(torch.load('../input/models/first_model.pt'))
deep_model.load_state_dict(torch.load('../input/deep-model/second_model.pt'))
if train_on_gpu:
    resnet_model.cuda() #.cuda() will move everything to the GPU side
    deep_model.cuda()
    
criterion = nn.CrossEntropyLoss()
criterion_deep = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(resnet_model.parameters(), lr=0.001,momentum=0.9,weight_decay=0.0005)
optimizer_deep= optim.SGD(deep_model.parameters(), lr=0.001,momentum=0.9,weight_decay=0.0005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
exp_lr_scheduler_deep = lr_scheduler.StepLR(optimizer_deep, step_size=7, gamma=0.1)
def train_model(resnet_model,deep_model,criterion, optimizer,optimizer_deep, scheduler,scheduler_deep, num_epochs=20):
    
    avg_loss_fisrt = 0
    avg_acc_first = 0
    avg_loss_second =0
    avg_acc_second =0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(traindataloaders)
    val_batches = len(validdataloaders)

    valid_loss_min = np.inf # track change in validation loss
    falsh_predicted_images_first  =[]
    falsh_predicted_labels_first  =[]
    falsh_predicted_images_second =[]
    falsh_predicted_labels_second =[]
    
    #train_images_data_without_dablicated = []
    #train_labels_data_without_dablicated = []
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print('-' * 10)
        
        loss_train_first = 0
        loss_train_second=0
        loss_val = 0
        
        acc_train_first = 0
        acc_train_second =0
        acc_val = 0
        
        resnet_model.train(True)
        deep_model.train(True)
        
        for i, data in enumerate(traindataloaders):
            if i % 10 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches ), end='', flush=True)
                
            inputs, labels = data
            inputs, labels = inputs.cpu().numpy(), labels.cpu().numpy()
            
            m = inputs.shape[0]
            inputs_first, labels_first = inputs[:int(m/2),:,:,:], labels[:int(m/2)]   # numpy
            inputs_second, labels_second = inputs[int(m/2):,:,:,:], labels[int(m/2):] # numpy
            inputs_first, labels_first = torch.from_numpy(inputs_first), torch.from_numpy(labels_first)
            inputs_second, labels_second = torch.from_numpy(inputs_second), torch.from_numpy(labels_second)

            if train_on_gpu :
                inputs_first, labels_first = inputs_first.cuda(), labels_first.cuda()
                inputs_second, labels_second = inputs_second.cuda(), labels_second.cuda()
            else:
                inputs_first, labels_first = inputs_first, labels_first     
                inputs_second, labels_second = inputs_second, labels_second
            
            if(np.array(falsh_predicted_images_second).shape[0]):
                inputs_first = torch.cat(((torch.from_numpy(np.array(falsh_predicted_images_second))).cuda(),inputs_first),dim=0)
                labels_first = torch.cat(((torch.from_numpy(np.array(falsh_predicted_labels_second))).cuda(),labels_first),dim=0) 
            optimizer.zero_grad()
            outputs_first = resnet_model(inputs_first)
            ps_first = torch.exp(outputs_first)
            top_p_first, top_class_first = ps_first.topk(1, dim=1)
            equals_first = top_class_first == labels_first.view(*top_class_first.shape)

            for i in range(len(equals_first)):
                if equals_first[i]==False :
                    falsh_predicted_images_first.append(inputs_first[i].cpu().numpy())
                    falsh_predicted_labels_first.append(labels_first[i].cpu().numpy()) 

            falsh_predicted_images_second[:] = []
            falsh_predicted_labels_second[:] = []
            if(np.array(falsh_predicted_images_first).shape[0]):
                inputs_second = torch.cat(((torch.from_numpy(np.array(falsh_predicted_images_first))).cuda(),inputs_second),dim=0)
                labels_second = torch.cat(((torch.from_numpy(np.array(falsh_predicted_labels_first))).cuda(),labels_second),dim=0)

            optimizer_deep.zero_grad()
            outputs_second = deep_model(inputs_second)

            ps_second = torch.exp(outputs_second)
            top_p_second, top_class_second = ps_second.topk(1, dim=1)
            equals_second = top_class_second == labels_second.view(*top_class_second.shape)
            for i in range(len(equals_second)):
                if equals_second[i]==False :
                    falsh_predicted_images_second.append(inputs_second[i].cpu().numpy())
                    falsh_predicted_labels_second.append(labels_second[i].cpu().numpy())      
                    
            falsh_predicted_images_first[:] = []
            falsh_predicted_labels_first[:] = []
            outputs = torch.cat((outputs_first,outputs_second),dim=0)
            labels  = torch.cat((labels_first,labels_second), dim=0)
            #ps = torch.exp(outputs)
            #ps_second = torch.exp(outputs_second)
            #top_p, top_class = ps.topk(1, dim=1)
            #top_p_second, top_class_second = ps_second.topk(1, dim=1)
            #print(top_class_second.shape,"top_class_second")
            #print(top_class_first.shape, "top_class_first")
            #equals = top_class == labels.view(*top_class.shape)
            equals = torch.cat((equals_first,equals_second), dim=0)
            #equals_second = top_class_second == labels_second.view(*top_class_second.shape)
            #equals = torch.cat((equals_first,equals_second),dim=0)
            #print(equals.shape)
            

            acc_train_first += torch.mean(equals.type(torch.FloatTensor)).item()
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer_deep.step()
            
            loss_train_first += loss.item() 

        resnet_model.train(False)
        deep_model.train(False)
        resnet_model.eval()
        deep_model.eval()
        with torch.no_grad() :
            
            for i, data in enumerate(validdataloaders):
                if i % 10 == 0:
                    print("\rValidation batch {}/{}".format(i+1, val_batches), end='', flush=True)
                inputs, labels = data
                inputs, labels = inputs.cpu().numpy(), labels.cpu().numpy()
            
                m = inputs.shape[0]
                inputs_first, labels_first = inputs[:int(m/2),:,:,:], labels[:int(m/2)]   # numpy
                inputs_second, labels_second = inputs[int(m/2):,:,:,:], labels[int(m/2):] # numpy
            
                inputs_first, labels_first = torch.from_numpy(inputs_first), torch.from_numpy(labels_first)
                inputs_second, labels_second = torch.from_numpy(inputs_second), torch.from_numpy(labels_second)                    
                
            
                if train_on_gpu :
                    inputs_first, labels_first = inputs_first.cuda(), labels_first.cuda()
                    inputs_second, labels_second = inputs_second.cuda(), labels_second.cuda()
                else:
                    inputs_first, labels_first = inputs_first, labels_first     
                    inputs_second, labels_second = inputs_second, labels_second

            
                optimizer.zero_grad()
                optimizer_deep.zero_grad()
                outputs_first = resnet_model(inputs_first)
                outputs_second = deep_model(inputs_second)
            
                outputs = torch.cat((outputs_first,outputs_second),dim=0)
                labels = torch.from_numpy(labels).cuda()
                ps = torch.exp(outputs)
                #ps_second = torch.exp(outputs_second)
                top_p, top_class = ps.topk(1, dim=1)
                #top_p_second, top_class_second = ps_second.topk(1, dim=1)
                #print(top_class_second.shape,"top_class_second")
                #print(top_class_first.shape, "top_class_first")
                equals = top_class == labels.view(*top_class.shape)
                #equals_second = top_class_second == labels_second.view(*top_class_second.shape)
                #equals = torch.cat((equals_first,equals_second),dim=0)
                #print(equals.shape)
                acc_val += torch.mean(equals.type(torch.FloatTensor))
                loss = criterion(outputs, labels)
            
                loss_val += loss.item()

        avg_loss_val = loss_val / len(validdataloaders)
        avg_acc_val = acc_val / len(validdataloaders)
        avg_loss_first = loss_train_first  /  len(traindataloaders)
        avg_acc_first = acc_train_first  / len(traindataloaders)
        
        #avg_loss_deep = loss_train_deep / len(mini_batches)
        #avg_acc_deep  = acc_train_deep / len(mini_batches)
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss_first))
        print("Avg acc (train): {:.4f}".format(avg_acc_first))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        #print("Avg loss (deep): {:.4f}".format(avg_loss_deep))
        #print("Avg acc (deep): {:.4f}".format(avg_acc_deep))
        print('-' * 10)
        print()
        if avg_loss_val <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            avg_loss_val))
            torch.save(resnet_model.state_dict(), 'first_model_v3.pt')
            torch.save(deep_model.state_dict(), 'second_model_v3.pt')
            valid_loss_min = avg_loss_val

    return resnet_model,deep_model
resnet_model,deep_model = train_model(resnet_model,deep_model, criterion, optimizer_ft,optimizer_deep, exp_lr_scheduler,exp_lr_scheduler_deep, num_epochs=20)
torch.save(resnet_model.state_dict(), 'first_model_v2.pt')
torch.save(deep_model.state_dict(), 'second_model_v2.pt')
def eval_model(resnet_model, criterion):
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    test_batches = len(validdataloaders)
    print("Evaluating model")
    print('-' * 10)
    with torch.no_grad():
        
        for i, data in enumerate(validdataloaders):
            if i % 10 == 0:
                print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

            resnet_model.train(False)
            resnet_model.eval()
            inputs, labels = data

            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

            outputs = resnet_model(inputs)

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            for i in equals :
                if not i:
                    print(labels)
                    print(top_class)
            acc_test += torch.mean(equals.type(torch.FloatTensor))
            loss = criterion(outputs, labels)
            

            loss_test += loss.item()

        avg_loss = loss_test / len(validdataloaders)
        avg_acc = acc_test / len(validdataloaders)
    
        print()
        print("Avg loss (test): {:.4f}".format(avg_loss))
        print("Avg acc (test): {:.4f}".format(avg_acc))
        print('-' * 10)
eval_model(resnet_model, criterion)
torch.save(resnet_model.state_dict(), 'resnet_model_final_PyTorch_Challenge.pt')
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(102))
class_total = list(0. for i in range(102))

resnet_model.eval()
# iterate over test data
for data, target in testdataloaders:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = resnet_model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(len(list(target.data.shape))):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(testdataloaders.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(102):
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
dataiter = iter(testdataloaders)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
output = resnet_model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 10))
for idx in np.arange(16):
    ax = fig.add_subplot(4, 8/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))

