import torch

import torch.nn as nn

import torch.nn.init as init

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data.sampler import *

from torch.utils.data import Dataset

from torchvision import transforms, datasets

import torchvision

import numpy as np

import torchvision.models as models



dataset_root = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/"

batch_size = 128

target_size = (224,224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get the transforms

def load_datasets():



    # Transforms for the image.

    transform = transforms.Compose([

                        transforms.Grayscale(3),

                        transforms.Resize(target_size),

                        transforms.ToTensor(),

                        transforms.Normalize((0.5,), (0.5,))

                ])



    # Define the image folder for each of the data set types

    trainset = torchvision.datasets.ImageFolder(

        root=dataset_root + 'train',

        transform=transform

    )

    validset = torchvision.datasets.ImageFolder(

        root=dataset_root + 'val',

        transform=transform

    )

    testset = torchvision.datasets.ImageFolder(

        root=dataset_root + 'test',

        transform=transform

    )





    # Define indexes and get the subset random sample of each.

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    valid_dataloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

        

    return train_dataloader, valid_dataloader, test_dataloader


# Transfer Learning Model for pneumonia

class TransferLearningModel(nn.Module):

    def __init__(self):

        super(TransferLearningModel, self).__init__()

        

        # Using VGG-16 as our transfer model

        self.prevmodel = models.vgg16(pretrained=True)



        # Not retraining the model. Instead, learning the output.

        # Freeze all layers we do not want the gradient to update.

        for param in self.prevmodel.parameters():

            param.require_grad = False



        # Delete last layer and add own final layer

        self.prevmodel.features[-1] = nn.Linear(14, 2)





    # The forward operation for the NN. Backward is auto computed.

    def forward(self, x):

        x = self.prevmodel(x)

        x = F.log_softmax(x)

        return x



# Following code appears at:  https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/

class One_Hot(nn.Module):

    def __init__(self, depth):

        super(One_Hot,self).__init__()

        self.depth = depth

        self.ones = torch.sparse.torch.eye(depth).to(device)

    def forward(self, X_in):

        X_in = X_in.long()

        return self.ones.index_select(0,X_in.data)



    def __repr__(self):

        return self.__class__.__name__ + "({})".format(self.depth)
# This probably exists already

def printif(message, isPrint):

    if isPrint:

        print(message)



# Test the model on the given dataset.

def test(model, dataset):

    model.eval()



    # Define loss, one_hot and variables

    mloss = nn.CrossEntropyLoss()

    one_hot = One_Hot(2).to(device)

    correct = 0

    loss_total = 0

    

    with torch.no_grad():

        for data, target in dataset:



            # Convert data and target to device

            data = data.to(device)

            target = target.to(device)



            # Get network output, loss and correct

            output = model(data)

            loss = mloss(output, target)

            _, pred = torch.max(output.data, dim=1)



            # Update correct and loss

            correct += pred.eq(target.data.view_as(pred)).sum().item()

            loss_total += loss.item()



    return 100. * correct / len(dataset.dataset), loss_total/ len(dataset.dataset)



def train(model, trainset, validset, learning_rate=0.001, decay=0, epochs=10, valid_interval=1, log_interval=10, console_logging=True):



    # Logging curves

    loss_curve = []

    accuracy_curve =[]

    validation_accuracy_curve = []

    validation_loss_curve = []

    

    # Loss and one hot

    mloss = nn.CrossEntropyLoss()

    one_hot = One_Hot(2).to(device)



    # Optimizer

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)



    max_accuracy = -1

    for epoch in range(1, epochs+1):



        epoch_loss = 0

        epoch_accuracy = 0



        for batch_idx, (data, target) in enumerate(trainset):

            model.train()



            # Convert data to device

            data = data.to(device)

            target = target.to(device).long()



            # Zerograd optimizer as pytorch will add gradients together if not cleared

            optimizer.zero_grad()



            # Get the output, calculate loss

            output = model(data)

            loss = mloss(output, target)



            epoch_loss += loss.item()



            # Compute the backwards update step, send it to optimizer

            loss.backward()

            optimizer.step()



            # Get correct items for logging

            _, pred = torch.max(output.data, dim=1)

            #print("Pred: " + str(pred))

            #print("Correct: " + str(target))

            correct = pred.eq(target.data.view_as(pred)).sum()

            epoch_accuracy += correct.item()



            if batch_idx % log_interval == 0:

                printif('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                                    epoch, batch_idx * batch_size, len(trainset.dataset),

                                    100. * (batch_idx * batch_size) / len(trainset.dataset), loss.item()), console_logging)

                

        # Add logged items to their curves

        loss_curve.append(epoch_loss / len(trainset.dataset))

        accuracy_curve.append(epoch_accuracy / len(trainset.dataset))



        # If we are a validation interval, validate and save best model.

        if epoch % valid_interval == 0:

            printif("Starting Validation Step:", console_logging)

            accuracy_val, loss_val = test(model, validset)

            validation_accuracy_curve.append(accuracy_val)

            validation_loss_curve.append(loss_val)

            printif("Validation Dataset - Accuracy: %.2f, Loss: %.4f" % (accuracy_val, loss_val), console_logging)

            if accuracy_val > max_accuracy:

                printif("New optimal validation value recieved!", console_logging)

                #torch.save(model, "/kaggle/output/model.pnt")

                max_accuracy = accuracy_val



    return loss_curve, accuracy_curve, validation_loss_curve, validation_accuracy_curve
trainset, valset, testset = load_datasets()

print("Loaded train, val and test with sizes: %d, %d, %d" % (len(trainset.dataset), len(valset.dataset), len(testset.dataset)))

model = TransferLearningModel().to(device)

print("Input Size: %d" % (target_size[0] * target_size[1]))

trainloss, trainaccuracy, validloss, validaccuracy = train(model, trainset, valset, epochs=10, decay=0.1, valid_interval=2, learning_rate=0.0001)
#lin = torch.load("/kaggle/output/model.pnt").to(device)

accuracy, loss = test(model, testset)

print("Test Dataset - Accuracy: %.2f, Loss: %.4f" % (accuracy, loss))

accuracy, loss = test(model, valset)

print("Valid Dataset - Accuracy: %.2f, Loss: %.4f" % (accuracy, loss))