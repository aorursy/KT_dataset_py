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

                        transforms.Grayscale(),

                        transforms.Resize(target_size),

                        transforms.ToTensor(),

                        transforms.Normalize((0.5,), (0.5,)),

                        nn.Flatten()

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

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)

    valid_dataloader = torch.utils.data.DataLoader(validset, batch_size=len(validset), shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)





    # Convert data to tensors. This could be made faster.

    x_test = []

    y_test = []

    for idx, (data, tar) in enumerate(test_dataloader):

        x_test = data.squeeze()

        y_test = tar.squeeze()



    x_train = []

    y_train = []

    for idx, (data, tar) in enumerate(train_dataloader):

        x_train = data.squeeze()

        y_train = tar.squeeze()

        

    x_test = torch.tensor(x_test, device=device)

    y_test = torch.tensor(y_test, device=device)

    x_train = torch.tensor(x_train, device=device)

    y_train = torch.tensor(y_train, device=device)

    return x_train, y_train, x_test, y_test
def knn(x_train, y_train, x_test, k, device, log_interval=100, log=True):



    # Get the amount of images, training images, and image size.

    num_images = x_test.shape[0]

    num_train = y_train.shape[0]

    img_size = x_test.shape[1]



    y_test = torch.zeros((num_images), device=device, dtype=torch.float)



    # For each of the images in the test set

    for test_index in range(0, num_images):



        # Get the image and calculate the distance to every item in the trainset

        test_image = x_test[test_index]

        distances = torch.norm(x_train - test_image, dim=1)



        # Get the top k indexes and get the most used index between them all

        indexes = torch.topk(distances, k, largest=False)[1]

        classes = torch.gather(y_train, 0, indexes)

        mode = int(torch.mode(classes)[0])



        # Save the test value in the index.

        y_test[test_index] = mode



        # Logging since with large sets it may be helpful

        if log:

            if test_index % log_interval == 0:

                print("Currently predicting at test_index = %d" % test_index)



    return y_test
print("Loading data from folders.")

x_train, y_train, x_test, y_test = load_datasets()

print("Loaded train and test with sizes: %s, %s" % (str(x_train.shape), str(x_test.shape)))
pred = knn(x_train, y_train, x_test, k=1, device=device)
correct = pred.eq(y_test.to(device).view_as(pred)).sum()

print("Correct predictions: %d/%d, Accuracy: %f" % (correct, y_test.shape[0], 100. * correct / y_test.shape[0]))
#https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm suggests k = ~sqrt(N).

k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 25, 37, 49, int(np.floor(np.sqrt(x_train.shape[0]))), 200]



correct_vals = []



best_k = -1

best_correct = 0



for k in k_values:

    pred = knn(x_train, y_train, x_test, k=k, device=device, log=False)

    correct = pred.eq(y_test.view_as(pred)).sum()

    print("K = %d, Correct: %d, Accuracy: %.2f" % (k, correct, 100. * correct / y_test.shape[0]))