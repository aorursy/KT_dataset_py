# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import torchvision

from torchvision import datasets, transforms

from torch import nn, optim

from torch.nn import functional as F

from torch.utils.data import DataLoader, sampler, random_split

from torchvision import models
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator

from ignite.handlers import ModelCheckpoint, EarlyStopping

from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix

from ignite.contrib.handlers import ProgressBar
def get_data_loaders(train_dir,test_dir, batch_size):

    transform = transforms.Compose([transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor()])

    train_data_all = datasets.ImageFolder(train_dir, transform=transform)

    test_data_all = datasets.ImageFolder(test_dir, transform=transform)

    valid_data_len = int(len(test_data_all) * 0.4)

    test_data_len = int(len(test_data_all) - valid_data_len)

    val_data, test_data = random_split(test_data_all, [valid_data_len, test_data_len])

    train_loader = DataLoader(train_data_all, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return ((train_loader, val_loader, test_loader), train_data_all.classes)
TRAIN_DIR = "/kaggle/input/intel-image-classification/seg_train/seg_train/"

TEST_DIR = "/kaggle/input/intel-image-classification/seg_test/seg_test/"

(train_loader, val_loader, test_loader), classes = get_data_loaders(TRAIN_DIR, TEST_DIR, 64)
print(classes)
print(len(train_loader))

print(len(val_loader))

print(len(test_loader))
dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))

    ax.set_title(classes[labels[idx]])
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

device
model = models.densenet201(pretrained=True)
print(model)
print(model.classifier.in_features) 

print(model.classifier.out_features)
for param in model.parameters():

    param.requires_grad = False
n_inputs = model.classifier.in_features

last_layer = nn.Linear(n_inputs, len(classes))

model.classifier = last_layer

if torch.cuda.is_available():

    model.cuda()

print(model.classifier.out_features)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters())
training_history = {'accuracy':[],'loss':[]}

validation_history = {'accuracy':[],'loss':[]}
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

evaluator = create_supervised_evaluator(model,

                                        device=device,

                                        metrics={

                                            'accuracy': Accuracy(),

                                            'loss': Loss(criterion),

                                            'cm':ConfusionMatrix(len(classes))

                                            })

@trainer.on(Events.ITERATION_COMPLETED)

def log_a_dot(engine):

    print(".",end="")



@trainer.on(Events.EPOCH_COMPLETED)

def log_training_results(trainer):

    evaluator.run(train_loader)

    metrics = evaluator.state.metrics

    accuracy = metrics['accuracy']*100

    loss = metrics['loss']

    training_history['accuracy'].append(accuracy)

    training_history['loss'].append(loss)

    print()

    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"

          .format(trainer.state.epoch, accuracy, loss))

    

@trainer.on(Events.EPOCH_COMPLETED)   

def log_validation_results(trainer):

    evaluator.run(val_loader)

    metrics = evaluator.state.metrics

    accuracy = metrics['accuracy']*100

    loss = metrics['loss']

    validation_history['accuracy'].append(accuracy)

    validation_history['loss'].append(loss)

    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"

          .format(trainer.state.epoch, accuracy, loss))

    

@trainer.on(Events.COMPLETED)

def log_confusion_matrix(trainer):

    evaluator.run(test_loader)

    metrics = evaluator.state.metrics

    cm = metrics['cm']

    cm = cm.numpy()

    cm = cm.astype(int)

    fig, ax = plt.subplots(figsize=(10,10))  

    ax= plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax,fmt="d")

    # labels, title and ticks

    ax.set_xlabel('Predicted labels')

    ax.set_ylabel('True labels') 

    ax.set_title('Confusion Matrix') 

    ax.xaxis.set_ticklabels(classes,rotation=90)

    ax.yaxis.set_ticklabels(classes,rotation=0)
trainer.run(train_loader, max_epochs=5)
fig, axs = plt.subplots(2, 2)

fig.set_figheight(6)

fig.set_figwidth(14)

axs[0,0].plot(training_history['accuracy'])

axs[0,0].set_title("Training Accuracy")

axs[0,1].plot(validation_history['accuracy'])

axs[0,1].set_title("Validation Accuracy")

axs[1,0].plot(training_history['loss'])

axs[1,0].set_title("Training Loss")

axs[1,1].plot(validation_history['loss'])

axs[1,1].set_title("Validation Loss")
test_loss = 0.0

class_correct = list(0. for i in range(len(classes)))

class_total = list(0. for i in range(len(classes)))



model.eval()



for data, target in test_loader:

    if torch.cuda.is_available():

        data, target = data.cuda(), target.cuda()

    output = model(data)

    loss = criterion(output, target)

    test_loss += loss.item()*data.size(0)

    _, pred = torch.max(output, 1)    

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())

    if len(target) == 64:

      for i in range(64):

          label = target.data[i]

          class_correct[label] += correct[i].item()

          class_total[label] += 1



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
dataiter = iter(test_loader)

images, labels = dataiter.next()

images.numpy()



# move model inputs to cuda, if GPU available

if torch.cuda.is_available():

    images = images.cuda()



# get sample outputs

output = model(images)

# convert output probabilities to predicted class

_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(preds_tensor.cpu().numpy())



# plot the images in the batch, along with predicted and true labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx].cpu(), (1, 2, 0)))

    ax.set_title("{} ({})".format(classes[preds[idx]],

                                  classes[labels[idx]]),

                 color=("green" if preds[idx]==labels[idx].item() else "red"))
from PIL import Image
import requests

from io import BytesIO
def apply_test_transforms(inp):

    out = transforms.functional.resize(inp, [224,224])

    out = transforms.functional.to_tensor(out)

    out = transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return out
model.eval()
def predict(model, filepath, show_img=False, url=False):

    if url:

        response = requests.get(filepath)

        im = Image.open(BytesIO(response.content))

    else:

         im = Image.open(filepath)

    if show_img:

        plt.imshow(im)

    im_as_tensor = apply_test_transforms(im)

    minibatch = torch.stack([im_as_tensor])

    if torch.cuda.is_available():

        minibatch = minibatch.cuda()

    pred = model(minibatch)

    _, classnum = torch.max(pred, 1)

    return classes[classnum]
predict(model, "https://assets.bwbx.io/images/users/iqjWHBFdfxIU/i.7bFb5aIp2g/v1/1000x-1.jpg", show_img=True, url=True)