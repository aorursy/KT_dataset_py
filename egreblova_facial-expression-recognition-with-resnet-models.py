import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.utils.data

import matplotlib.pyplot as plt



# set all seed to 0

random.seed(0)

np.random.seed(0)

torch.manual_seed(0)

torch.cuda.manual_seed(0)

torch.backends.cudnn.deterministic = True
# read data

all_data = pd.read_csv("../input/fer2013/fer2013.csv")
# split to 3 parts

groups = [g for _, g in all_data.groupby('Usage')]

training_data = groups[2]

validation_data = groups[1]

testing_data = groups[0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def make_dataloader(data, batch_size, shuffle):

    images, labels = data['pixels'], data['emotion']

    images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in images]) / 255.0 # normalizing data to be between 0 and 1

    images = torch.FloatTensor(images.reshape(images.shape[0], 1, 48, 48)).to(device) # 1 color channel, 48x48 images

    dataset = torch.utils.data.TensorDataset(images, torch.LongTensor(np.array(labels)).to(device))

    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
train_loader = make_dataloader(training_data, 100, True)

valid_loader = make_dataloader(validation_data, 100, False)
dataiter = iter(train_loader)

images, labels = dataiter.next()

print(label_names[labels[1]])

plt.imshow(images[1].view(48, 48).cpu());
import torch.nn as nn

def adjust_model(model):

    model.conv1 = nn.Conv2d(1, 64, model.conv1.kernel_size, model.conv1.stride, model.conv1.padding, bias=False)

    model.fc = nn.Linear(model.fc.in_features, 7, bias=False)

    return model
epochs = 100
def eval_model(model, data_loader, criterion):

    model.eval()

    with torch.no_grad():

        accuracy = 0

        loss = 0

        for data, labels in data_loader:

            output = model(data)

            _, preds = torch.max(output.data, 1)

            equals = (preds == labels).cpu()

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            loss += criterion(output, labels).data.cpu()

        return accuracy/len(data_loader), loss/len(data_loader)

        

def train_model(model, criterion, optimizer, data_loader, eval_loader):

    model = model.to(device)

    test_accuracy_history = []

    test_loss_history = []

    for epoch in range(epochs):

        model.train()

        for data, labels in data_loader:

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            

        accuracy, loss = eval_model(model, eval_loader, criterion)

        test_accuracy_history.append(accuracy)

        test_loss_history.append(loss)

    return test_accuracy_history, test_loss_history
from torch import optim

from torchvision import models

criterion = nn.CrossEntropyLoss()

lrs = [0.1, 0.01, 0.001, 0.0001, 0.000001]

models_1 = [adjust_model(models.resnet18()) for i in range(len(lrs))]

optimizers = [optim.SGD(models_1[i].parameters(), lr=lrs[i], momentum=0.9) for i in range(len(lrs))]
for i in range(len(lrs)):

    accuracy, loss = train_model(models_1[i], criterion, optimizers[i], train_loader, valid_loader)

    torch.save(accuracy, 'ResNet18_lr_'+ str(lrs[i]) + '_accuracy.pt')

    torch.save(loss, 'ResNet18_lr_'+ str(lrs[i]) + '_loss.pt') 

    torch.save(models_1[i], 'ResNet18_lr_'+ str(lrs[i]) + '_model.pt')
resnet18_accuracy = [torch.load("../input/fer2013-results/results/ResNet18_lr_0.1_accuracy.pt"), 

            torch.load("../input/fer2013-results/results/ResNet18_lr_0.01_accuracy.pt"),

            torch.load("../input/fer2013-results/results/ResNet18_lr_0.001_accuracy.pt"),

            torch.load("../input/fer2013-results/results/ResNet18_lr_0.0001_accuracy.pt"),

            torch.load("../input/fer2013-results/results/ResNet18_lr_1e-06_accuracy.pt")]



resnet18_loss = [torch.load("../input/fer2013-results/results/ResNet18_lr_0.1_loss.pt"),

                 torch.load("../input/fer2013-results/results/ResNet18_lr_0.01_loss.pt"),

                 torch.load("../input/fer2013-results/results/ResNet18_lr_0.001_loss.pt"),

                 torch.load("../input/fer2013-results/results/ResNet18_lr_0.0001_loss.pt"),

                 torch.load("../input/fer2013-results/results/ResNet18_lr_1e-06_loss.pt")]
colors = ['skyblue', 'red', 'green', 'violet', 'magenta']



def make_plots(accuracy, losses, title1, title2, lbls):

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)

    ax1.title.set_text(title1)

    for i in range(len(accuracy)):

        ax1.plot(range(epochs),accuracy[i], color=colors[i],label=lbls[i])

    ax1.set_xlabel('epochs');

    ax1.set_ylabel('accuracy')

    ax1.legend(loc='lower right')



    ax2 = fig.add_subplot(1, 2, 2)

    ax2.title.set_text(title2)

    for i in range(len(losses)):

        ax2.plot(range(epochs),losses[i], color=colors[i],label=lbls[i])

    ax2.set_xlabel('epochs');

    ax2.set_ylabel('loss')

    ax2.legend(loc='upper left')

    plt.subplots_adjust(wspace=0.35, right=2.0)

    plt.show()
title1 = 'ResNet18 accuracy with different larning rates'

title2 = 'ResNet18 loss with different larning rates'

make_plots(resnet18_accuracy, resnet18_loss, title1, title2, [str(lr) for lr in lrs])
def print_acc_loss_results(name_tag, var_s, accs, losses):

    min_inds = []

    print("Min losses:")

    for i in range (len(losses)):

        min_v = min(losses[i])

        min_ind = losses[i].index(min_v)

        min_inds.append(min_ind)

        print(name_tag + ' {}: Loss {:.5f} at the epoch # {}'.format(var_s[i],min_v, min_ind+1))

    print("\nResulting accuracies:")

    for i in range (len(accs)):

        print(name_tag + ' {}: Accuracy {:.2f}'.format(var_s[i],np.mean(accs[i][-10:-1]) * 100.0))

    print("\nAccuracies at minimal loss epoch:")

    for i in range (len(accs)):

        print(name_tag + ' {}: Accuracy {:.2f} at the epoch # {}'.format(var_s[i],accs[i][min_inds[i]] * 100, min_inds[i]+1))
print_acc_loss_results('Learning rate', lrs, resnet18_accuracy, resnet18_loss)
models_2 = [adjust_model(models.resnet50()), adjust_model(models.resnet152())]

optimizers_2 = [optim.SGD(models_2[i].parameters(), lr=0.1, momentum=0.9) for i in range(2)]

names = ['ResNet50', 'ResNet152']
for i in range(len(names)):

    accuracy, loss = train_model(models_2[i], criterion, optimizers_2[i], train_loader, valid_loader)

    torch.save(accuracy, names[i] + '_0.1_accuracy.pt')

    torch.save(loss, names[i] + '_0.1_loss.pt') 

    torch.save(models_2[i], names[i] + '_0.1_model.pt')
resnet50_152_accuracy = [torch.load("../input/fer2013-results/results/ResNet50_0.1_accuracy.pt"), 

            torch.load("../input/fer2013-results/results/ResNet152_0.1_accuracy.pt")]



resne50_152_loss = [torch.load("../input/fer2013-results/results/ResNet50_0.1_loss.pt"),

                 torch.load("../input/fer2013-results/results/ResNet152_0.1_loss.pt")]
title1 = 'Accuracy for ResNet50 and ResNet152'

title2 = 'Loss for ResNet50 and ResNet152'

make_plots(resnet50_152_accuracy, resne50_152_loss, title1, title2, names)
print_acc_loss_results('Model', names, resnet50_152_accuracy, resne50_152_loss)
test_loader = make_dataloader(testing_data, 100, False)

model_best = torch.load('../input/fer2013-results/results/ResNet18_lr_0.1_model.pt')

acc, losses = eval_model(model_best, test_loader, criterion)
print('Test accuracy: ' + str(acc*100))