from shutil import copyfile

copyfile(src = "../input/modules/dataset_det.py", dst = "../working/dataset_det.py")

copyfile(src = "../input/vis-module/vis.py", dst = "../working/vis.py")
import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy

from collections import OrderedDict

import matplotlib.pyplot as plt



import dataset_det as d_loader

!pip install ipdb

import vis

import torch.utils.data as torch_d

from torch.nn import functional as F

!pip install torchsummary

from torchsummary import summary
BATCHSIZE=50



dataset = d_loader.Balls_CF_Detection ("../input/balls-images/train", 21000)

train_dataset, test_dataset = torch_d.random_split(dataset, [int(21000*0.9), int(21000*0.1)])



dataloaders = {}

dataloaders['train'] = torch.utils.data.DataLoader(train_dataset,batch_size=BATCHSIZE, shuffle=True)

dataloaders['val'] = torch.utils.data.DataLoader(test_dataset,batch_size=BATCHSIZE, shuffle=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']

first_batch = next(iter(dataloaders["train"]))

images, labels, bb = first_batch

images = images.to(device)



for i in range(5):

    plt.imshow(np.asarray(vis.show_bboxes(images[i].cpu(), bb[i].cpu(), COLORS)))

    plt.show()
vgg16 = models.vgg16(pretrained=True)



for param in vgg16.features.parameters():

    param.requires_grad = False



num_features = vgg16.classifier[0].in_features



#features = list(vgg16.classifier.children())[:-1] # Remove last layer

#features.extend([nn.Linear(num_features, 36)]) # Add our layer with 9 outputs

#vgg16.classifier = nn.Sequential(*features) # Replace the model classifier



vgg16.classifier = nn.Sequential(

        nn.Linear(num_features, 500),

        nn.ReLU(),

        #nn.Dropout(0.5),

        nn.Linear(500, 36),

        )



print(next(iter(vgg16.features[0].parameters())).requires_grad)

print(vgg16)
def intersect(box_a, box_b):

    """ We resize both tensors to [A,B,2] without new malloc:

    [A,2] -> [A,1,2] -> [A,B,2]

    [B,2] -> [1,B,2] -> [A,B,2]

    Then we compute the area of intersect between box_a and box_b.

    Args:

      box_a: (tensor) bounding boxes, Shape: [A,4].

      box_b: (tensor) bounding boxes, Shape: [B,4].

    Return:

      (tensor) intersection area, Shape: [A,B].

    """

    A = box_a.size(0)

    B = box_b.size(0)

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),

                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),

                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)

    return inter[:, :, 0] * inter[:, :, 1]
def IOU(box_a, box_b):

    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap

    is simply the intersection over union of two boxes.  Here we operate on

    ground truth boxes and default boxes.

    E.g.:

        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    Args:

        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]

        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]

    Return:

        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]

    """

    inter = intersect(box_a, box_b)

    area_a = ((box_a[:, 2]-box_a[:, 0]) *

              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]

    area_b = ((box_b[:, 2]-box_b[:, 0]) *

              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter

    return inter / union  # [A,B]
def calcLoss (model, dataloader, customLoss, mse):

    correct = 0

    with torch.no_grad():

        for data in dataloader:

            images, labels, bb= data

            images = images.to(device)

            bb = bb.to(device)

            outputs = model(images)

            correct += (customLoss(outputs, bb) + mse(outputs, bb.view(-1, 36))) * outputs.size(0)

    return (correct / len(dataloader.dataset))
def iouLoss(outputs, groundtruth_bb):

    bb = groundtruth_bb.view(-1, 36)

    iou = 0

    bb = bb.view(-1, 9, 4)

    outputs = outputs.view(-1, 9, 4)

    for i in range(outputs.size(0)):

        iou += IOU(outputs[i], bb[i])

    iou = iou / outputs.size(0)

    return (1 - iou.mean())
def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):

    train_accuracies = []

    valid_accuracies = []

    

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = calcLoss(model, dataloaders["val"], customLoss, criterion)

    print('Initial val loss: {:.4f}'.format(best_loss))



    for epoch in range(1, num_epochs+1):

        print('Epoch {}/{}'.format(epoch, num_epochs))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels, bb in dataloaders[phase]:

                inputs = inputs.to(device)

                bb = bb.view(-1, 36).to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    

                    loss = customLoss(outputs, bb) + criterion(outputs, bb)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

            if phase == 'train' and scheduler != None:

                scheduler.step()



            epoch_loss = running_loss / (len(dataloaders[phase].dataset))

            

            if phase == 'train':

                train_accuracies.append(epoch_loss)

            else:

                valid_accuracies.append(epoch_loss)



            print('{} Loss: {:.4f} '.format(

                phase, epoch_loss))



            # deep copy the model

            if phase == 'val' and epoch_loss < best_loss:

                best_loss = epoch_loss

                best_model_wts = copy.deepcopy(model.state_dict())



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model, train_accuracies, valid_accuracies
model = vgg16.to(device)



criterion = torch.nn.MSELoss()

customLoss = iouLoss



optimizer = optim.Adam(model.parameters(), 0.00001)
EPOCH_NUMBER = 70



model, train_accs, val_accs = train_model(dataloaders, model, criterion, optimizer, None,

                       num_epochs=EPOCH_NUMBER)



torch.save(model.state_dict(), "./model")
f = plt.figure(figsize=(10, 8))

plt.plot(train_accs, label='training loss')

plt.plot(val_accs, label='validation loss')

plt.legend()

plt.show()
COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']

first_batch = next(iter(dataloaders["val"]))

images, labels, bb = first_batch

images = images.to(device)

preds_bb = model(images).view(-1, 9, 4)



for i in range(5):

    plt.imshow(np.asarray(vis.show_bboxes(images[i].cpu(), preds_bb[i].cpu(), COLORS)))

    plt.show()