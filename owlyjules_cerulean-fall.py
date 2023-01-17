import numpy as np

import matplotlib.pyplot as plt

import random

import os

import time

from sklearn.metrics import classification_report, confusion_matrix



import torch

import torchvision

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



def pr(x):

    print(repr(x))



def show(img):

    img = img / 2 + 0.5 # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()



def show_random_img(data):

    random_img = data[random.randint(0, len(data) - 1)]

    show(random_img[0])

    print('This is a {}'.format(get_classes()[random_img[1]]))



def get_classes():

    return sorted(os.listdir('../input/seg_train/seg_train/'))



def loaderify(data, train):

    batch_size = 16 if train else 1

    shuffle = True

    num_workers = 2

    

    loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

    

    return loader



def get_data(type_):

    transform = transforms.Compose([

        transforms.Resize(size = (150, 150)),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

    

    data = torchvision.datasets.ImageFolder('../input/seg_{0}/seg_{0}/'.format(type_), transform = transform)

    

    return data



class Convnet(nn.Module):

    def __init__(self, layer_one_neurons = 1000, layer_two_neurons = 100):

        super(Convnet, self).__init__()

        

        self.layer_one_neurons = layer_one_neurons

        self.layer_two_neurons = layer_two_neurons

        

        self.pool = nn.MaxPool2d(2, 2)

        

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3)

        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 3)

        

        self.magic_number = 25920 # I don't know how to calculate this manually Q_Q

        

        self.lin1 = nn.Linear(self.magic_number, self.layer_one_neurons)

        self.lin2 = nn.Linear(self.layer_one_neurons, self.layer_two_neurons)

        self.lin3 = nn.Linear(self.layer_two_neurons, 6)



    def forward(self, x):

        x = self.conv1(x)

        x = F.relu(x)

        x = self.pool(x)

        

        x = self.conv2(x)

        x = F.relu(x)

        x = self.pool(x)

        

        x = x.view(-1, self.magic_number)

        

        x = self.lin1(x)

        x = F.relu(x)

        

        x = self.lin2(x)

        x = F.relu(x)

        

        x = self.lin3(x)

        

        return x



# class Convnet(nn.Module):

#     def __init__(self, layer_one_neurons = 1000, layer_two_neurons = 100):

#         super(Convnet, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3)

#         self.pool = nn.MaxPool2d(2, 2)

#         self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 3)

        

#         self.layer_one_neurons = layer_one_neurons

#         self.layer_two_neurons = layer_two_neurons

        

#         self.lin1 = nn.Linear(25920, self.layer_one_neurons)

#         self.lin2 = nn.Linear(self.layer_one_neurons, self.layer_two_neurons)

#         self.lin3 = nn.Linear(self.layer_two_neurons, 6)



#     def forward(self, x):

#         x = self.conv1(x)

#         x = F.relu(x)

#         x = self.pool(x)

        

#         x = self.conv2(x)

#         x = F.relu(x)

#         x = self.pool(x)

        

#         x = x.view(-1, 25920)

        

#         x = self.lin1(x)

#         x = F.relu(x)

        

#         x = self.lin2(x)

#         x = F.relu(x)

        

#         x = self.lin3(x)

        

#         return x



def train(convnet, loader, epochs = 1):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(convnet.parameters(), lr = 0.01, momentum = 0.9)

    

    for epoch in range(epochs):

        running_loss = 0.0

        for i, data in enumerate(loader, 0):

            inputs, labels = data



            optimizer.zero_grad()



            outputs = convnet(inputs.cuda())

            loss = criterion(outputs, labels.cuda())

            loss.backward()

            optimizer.step()



            running_loss += loss.item()

            

            every_n = 300

            if i % every_n == every_n - 1:

                print('[%d, %5d] loss: %.3f' %

                      (epoch + 1, i + 1, running_loss / every_n))

                running_loss = 0.0



def classify_random_images(convnet, test_loader):

    dataiter = iter(test_loader)

    classes = get_classes()

    

    for i in range(10):

        images, labels = dataiter.next()

        

        show(images[0])

        

        outputs = convnet(images.cuda())

        _, predicteds = torch.max(outputs, 1)

        predicted = classes[predicteds[0]]

        groundtruth = classes[labels[0]]

        

        print('Prediction: {}\nGround truth: {}'.format(predicted, groundtruth))



def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):

    """pretty print for confusion matrixes"""

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length

    empty_cell = " " * columnwidth

    

    # Begin CHANGES

    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "

    

    if len(fst_empty_cell) < len(empty_cell):

        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell

    # Print header

    print("    " + fst_empty_cell, end=" ")

    # End CHANGES

    

    for label in labels:

        print("%{0}s".format(columnwidth) % label, end=" ")

        

    print()

    # Print rows

    for i, label1 in enumerate(labels):

        print("    %{0}s".format(columnwidth) % label1, end=" ")

        for j in range(len(labels)):

            cell = "%{0}.1f".format(columnwidth) % cm[i, j]

            if hide_zeroes:

                cell = cell if float(cm[i, j]) != 0 else empty_cell

            if hide_diagonal:

                cell = cell if i != j else empty_cell

            if hide_threshold:

                cell = cell if cm[i, j] > hide_threshold else empty_cell

            print(cell, end=" ")

        print()



def get_predictions(convnet, data_loader):

    y_true = []

    y_pred = []

    

    for data in data_loader:

        images, labels = data

        images = images.cuda()

        labels = labels.cuda()

        

        outputs = convnet(images)

        _, predicted = torch.max(outputs.data, 1)

        y_true += labels.tolist()

        y_pred += predicted.tolist()

    

    return [y_true, y_pred]



def print_metrics(convnet, test_loader):

    y_true, y_pred = get_predictions(convnet, test_loader)

    

    classes = get_classes()

    

    print('\nclassification report:')

    print(classification_report(y_true, y_pred, target_names = classes))

    

    y_true = [classes[a] for a in y_true]

    y_pred = [classes[a] for a in y_pred]

    

    cm = confusion_matrix(y_true, y_pred, labels = classes)

    print('\nconfusion matrix:')

    print_cm(cm, classes)



def seal_your_fate():

    seed = 2020

    

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)



def get_data_loaders():

    train_data = get_data('train')

    test_data = get_data('test')

    

    val_size = int(len(test_data) * 2 / 3)

    train_size = len(train_data) - val_size

    

    train_data_split, val_data_split = torch.utils.data.random_split(train_data, [train_size, val_size])

    

    train_loader = loaderify(train_data_split, train = True)

    val_loader = loaderify(val_data_split, train = False)

    test_loader = loaderify(test_data, train = False)

    

    return [train_loader, val_loader, test_loader]



def evaluate(convnet, test_loader):

    y_true, y_pred = get_predictions(convnet, test_loader)

    

    correct = (np.array(y_true) == np.array(y_pred)).sum()

    total = len(y_true)

    evaluation = correct / total

    

    return evaluation



def convnet_to_string(convnet):

    return 'Convnet({}, {})'.format(convnet.layer_one_neurons, convnet.layer_two_neurons)



def play_around(data_loaders):

    train_loader, val_loader, test_loader = data_loaders

    

    training_params = [[1000, 100], [12000, 2000], [6000, 200]]

    

    convnets = [Convnet(layer_one_neurons, layer_two_neurons) for (layer_one_neurons, layer_two_neurons) in training_params]

    

    for convnet in convnets:

        convnet.cuda()

        

        start = time.time()

        print('{} is training...\n'.format(convnet_to_string(convnet)))



        train(convnet, train_loader, epochs = 30)



        end = time.time()

        print('\n{} finished training in {:.2f} seconds\n'.format(convnet_to_string(convnet), end - start))

    

    print('Evaluating the convnets using the validation data...\n')

    convnets = [[convnet, evaluate(convnet, val_loader)] for convnet in convnets]

    

    for (convnet, score) in convnets:

        print('{} has a score of {:.3f}'.format(convnet_to_string(convnet), score))

    

    best_convnet = max(convnets, key = lambda x: x[1])[0]

    

    print('\nBest convnet: {}'.format(convnet_to_string(best_convnet)))

    

    classify_random_images(best_convnet, test_loader)

    print_metrics(best_convnet, test_loader)



def main():

    seal_your_fate()

    data_loaders = get_data_loaders()

    play_around(data_loaders)



main()

print('\ndone')