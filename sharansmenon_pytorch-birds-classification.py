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
!tar -xvzf /kaggle/input/200-bird-species-with-11788-images/CUB_200_2011.tgz
import torch

import torchvision

from torchvision import datasets, transforms

from torch import nn, optim

from torch.nn import functional as F

from torch.utils.data import DataLoader, sampler, random_split
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
import numpy as np

from torchvision import models
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator

from ignite.handlers import ModelCheckpoint, EarlyStopping

from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix

from ignite.contrib.handlers import ProgressBar
def get_data_loaders(data_dir, batch_size):

  transform = transforms.Compose([transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor()])

  all_data = datasets.ImageFolder(data_dir, transform=transform)

  train_data_len = int(len(all_data)*0.75)

  valid_data_len = int((len(all_data) - train_data_len)/2)

  test_data_len = int(len(all_data) - train_data_len - valid_data_len)

  train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

  return ((train_loader, val_loader, test_loader), all_data.classes)
(train_loader, val_loader, test_loader), classes = get_data_loaders("CUB_200_2011/images/", 64)
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
model = models.densenet169(pretrained=True)
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

optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
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
trainer.run(train_loader, max_epochs=7)
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
from PIL import Image

from io import BytesIO

import requests
def apply_test_transforms(inp):

    out = transforms.functional.resize(inp, [224,224])

    out = transforms.functional.to_tensor(out)

    out = transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return out
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

    print(classnum)

    return classes[classnum]
def formatText(string):

    string = string[4:]

    string = string.replace("-", " ")

    return string
formatText(predict(model, "https://www.audubon.org/sites/default/files/styles/hero_cover_bird_page/public/web_h_andy-morffew_flickr-creative-common-by-2.0_altamiraoriole_flickr-3-adult.jpg?itok=ad9rnLPN", show_img=True, url=True))
formatText(predict(model, "CUB_200_2011/images/080.Green_Kingfisher/Green_Kingfisher_0027_71048.jpg", show_img=True))
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

fig = plt.figure(figsize=(15, 10))

for idx in np.arange(10):

    ax = fig.add_subplot(10, 2, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx].cpu(), (1, 2, 0)))

    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),

                 color=("green" if preds[idx]==labels[idx].item() else "red"))
x = torch.randn(1, 3, 224, 224, requires_grad=False).to(device)
example = torch.rand(1, 3, 224, 224)

traced_script_module = torch.jit.trace(model.cpu(), example)
traced_script_module.save("birds-classification.zip")