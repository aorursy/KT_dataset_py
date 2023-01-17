import torch

import torchvision

import tensorflow as tf

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from torch import nn, cuda, optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

from torch.utils.data.sampler import SubsetRandomSampler



from sklearn.metrics import confusion_matrix



%config InlineBackend.figure_format = 'svg'

import seaborn as sns

sns.set()
def dataSampler(imageDir, trainSamples, testSamples, batchSize):

    '''

      Overview:

          Loads data and splits into training and test sets, also sets up

          batch size for training a model.



      Inputs:

          dir - Location of training images.

          trainSamples - Number of training samples from each category.

          testSamples - Number of test samples from each category.

          batchSize - Size of batches produced.



      Returns:

          Data loaders for training and validation data.

      '''

    

    numWorkers = 4

    



    # Transformations

    trainTransforms = transforms.Compose([

      transforms.Resize(256),

      transforms.RandomAffine(degrees=(-20, 20), translate=(0.2, 0.2), scale=(0.5, 2)),

      transforms.CenterCrop(size=224),

      transforms.ToTensor(),

      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])



    valTransforms = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

      ])



    # Load files into image folders.

    trainData = datasets.ImageFolder(imageDir, transform=trainTransforms)

    valData = datasets.ImageFolder(imageDir, transform=valTransforms)



    # Number of images in each category.

    catImages = 1000



    # Dictionary of image categories.

    cat = trainData.class_to_idx 



    trainIdx = []

    testIdx = []



    for i in range(len(cat)):



        n = i * catImages



        trainIdx += list(range(n, n+trainSamples))

        testIdx += list(range(n+trainSamples, n+trainSamples+testSamples))



    # Load images with batches and random samples.

    trainLoader = torch.utils.data.DataLoader(trainData,

                                              batchSize,

                                              sampler=SubsetRandomSampler(trainIdx),

                                              num_workers=numWorkers)



    valLoader = torch.utils.data.DataLoader(valData,

                                            batchSize,

                                            sampler=SubsetRandomSampler(testIdx),

                                            num_workers=numWorkers)



    return trainLoader, valLoader
def modelSetup():

    '''

    Overview:

        Loads a pretrained model and adds classifier for the dataset.

    

    Returns:

        model - A pretrained modified model.

    '''

    

    model = models.alexnet(pretrained=True)



    for param in model.parameters():

        param.requires_grad = False



    n_inputs = 4096

    n_classes = 40

    model.classifier[6] = nn.Sequential(

                          nn.Linear(n_inputs, 256), 

                          nn.ReLU(), 

                          nn.Dropout(0.4),

                          nn.Linear(256, n_classes),                   

                          nn.LogSoftmax(dim=1))



    model.cuda()



    return model
def plotTrainingResults(tLoss, vLoss, vAcc, n):

    '''

    Overview:

        Plots a graph displaying the loss and accuracy over iterations.

    

    Inputs:

        tLoss - Training loss history array.

        vLoss - Validation loss history array.

        vAcc - Validation accuracy history array.

        n - Gap between iteration validations.

    '''



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))



    # Loss graph.

    ax1.set_title('Loss')

    ax1.set_ylim((0, 4))

    ax1.set_xlabel('Iterations')

    ax1.set_ylabel('Loss')

    ax1.plot(tLoss, label='Training')

    ax1.plot(range(0, len(tLoss), n), vLoss, label='Validation')

    ax1.legend()



    # Accuracy graph.

    ax2.set_title('Accuracy')

    ax2.set_xlabel('Iterations')

    ax2.set_ylabel('Accuracy %')

    ax2.plot(range(0, len(tLoss), n), vAcc, label='Validation')

    ax2.legend()



    plt.show()
def trainModel(model, trainLoader, valLoader, validateEvery, epochs, learningRate):

    '''

    Overview:

        This is a function to train a CNN model using new data

    

    Inputs:

        model - The model to train on.

        validateEvery - The number of training iterations before performing validation step.

        epochs - The number of epochs.

    

    Returns:

        Trained model.

    '''



    device = torch.device('cuda')



    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=learningRate)



    print('Training model...')

    

    valAccHist = []

    trainLossHist = []

    valLossHist = []

    

    counter = 0



    for epoch in range(epochs):



        print('\nEpoch: {}'.format(epoch))



        # Training the model

        for inputs, labels in trainLoader:



            model.train()



            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            trainLossHist += [loss.item()]



            # Validate model every n iterations.

            if counter % validateEvery == 0:



                valLoss = 0

                valAcc = 0



                # Validating the model

                model.eval()

                with torch.no_grad():

                    for inputs, labels in valLoader:



                        inputs, labels = inputs.to(device), labels.to(device)

                        output = model.forward(inputs)

                        valLoss += criterion(output, labels).item()



                        output = torch.exp(output)

                        top_p, top_class = output.topk(1, dim=1)

                        equals = top_class == labels.view(*top_class.shape)

                        valAcc += torch.mean(equals.type(torch.FloatTensor)).item()





                # Output statistics.

                valAccHist += [valAcc / len(valLoader)]

                valLossHist += [valLoss / len(valLoader)]



                print('Iter: {} \tVal Accuracy: {:.6f} \tTrain Loss: {:.6f} \tVal Loss: {:.6f}'.format(counter, valAcc/len(valLoader), loss.item(), valLoss/len(valLoader)))



            counter += 1

                

    print('\nModel complete.')

    

    plotTrainingResults(trainLossHist, valLossHist, valAccHist, validateEvery)

    

    return model
def validateModel(model, valLoader):

    '''

    Overview:

        Predicts a validation set using the CNN model and prints accuracy values.

        

    Inputs:

        model - Machine learning model.

        valLoader - Validation set.

    '''

    

    device = torch.device('cuda')

    valLoss = 0

    val1Acc = 0

    val5Acc = 0

    

    # Confusion matrix lists

    pred = []

    true = []

    

    criterion = nn.NLLLoss()

    

    print('Model Validation.')

    

    model.eval()

    with torch.no_grad():

        for inputs, labels in valLoader:

            

            # Calculates loss value.

            inputs, labels = inputs.to(device), labels.to(device)

            output = model.forward(inputs)

            valLoss += criterion(output, labels).item()



            # Calculates validation top 1 accuracy.

            output = torch.exp(output)

            _, top_class = output.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)

            val1Acc += torch.mean(equals.type(torch.FloatTensor)).item()

            

            true += labels.tolist()

            pred += top_class.flatten().tolist()

            

            # Calculates validation top 5 accuracy.

            _, top_class = output.topk(5, dim=1)

            equals = top_class == labels.view(labels.shape[0], 1)

            val5Acc += torch.sum(equals).item() / labels.shape[0]

            

    # Calculates average.

    n = len(valLoader)

    valLoss /= n

    val1Acc /= n

    val5Acc /= n

    

    print('Loss: {:.6f} \t Top-1 Accuracy: {:.6f} \t Top-5 Accuracy: {:.6f}\n'.format(valLoss, val1Acc, val5Acc))

    

    # Confusion matrix

    cm = confusion_matrix(true, pred)

    plt.figure(figsize=(12, 9))

    plt.title('Confusion Matrix')

    sns.heatmap(cm)
def run(batchSize, epochs, learningRate):

    '''

    Overview:

        A helper function to create models in order to find the best hyperparameters.

    

    '''

    imageDir = '/kaggle/input/training-images/training_images'

    trainLoader, valLoader = dataSampler(imageDir, 400, 100, batchSize)

    model = modelSetup()

    model = trainModel(model, trainLoader, valLoader, 50, epochs, learningRate)
run(batchSize=150, epochs=7, learningRate=0.0001)
imageDir = '/kaggle/input/training-images/training_images'

trainLoader, valLoader = dataSampler(imageDir, 800, 200, 100)

model = modelSetup()

model = trainModel(model, trainLoader, valLoader, 50, 10, 0.001)
validateModel(model, valLoader)
# Class labels.

trainLoader.dataset.class_to_idx
imageDir = '/kaggle/input/test-images/testset'

testTransforms = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

      ])



testData = datasets.ImageFolder(imageDir, transform=testTransforms)

testLoader = torch.utils.data.DataLoader(testData, 100)



validateModel(model, testLoader)