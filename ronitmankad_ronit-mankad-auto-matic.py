import matplotlib.pyplot as plt

import numpy as np

import plotly.offline as py

import plotly.graph_objs as go

import seaborn as sns



from sklearn.metrics import roc_auc_score

 

import cv2

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, sampler, TensorDataset, random_split

from torch.optim import lr_scheduler

from torchvision import datasets, transforms, models

from torch import utils



import  time, copy, glob, torchvision, torch, os, json, re

from collections import Counter, OrderedDict



from PIL import Image

from sklearn.metrics import classification_report





plt.rcParams['figure.figsize']=(20,10)

py.init_notebook_mode(connected=False)

%matplotlib inline

print(os.listdir('../input'))
ROOT_DIR = '../input/virtual-hack/car_data/car_data'

TRAIN_DIR = '../input/virtual-hack/car_data/car_data/train'

TEST_DIR = '../input/virtual-hack/car_data/car_data/test'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 196
train_transforms = transforms.Compose([

    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),

    transforms.RandomRotation(degrees=15),

    transforms.RandomHorizontalFlip(),

    transforms.CenterCrop(size=224),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406],

                         [0.229, 0.224, 0.225])])



valid_transforms = transforms.Compose([transforms.CenterCrop(size=224),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406],

                         [0.229, 0.224, 0.225])])



dataset = datasets.ImageFolder(TRAIN_DIR, train_transforms)

test_dataset = datasets.ImageFolder(TEST_DIR, valid_transforms)
size = len(dataset)

val_split = 0.2

trainset, valset = random_split(dataset, [size - int(size * val_split), int(size * val_split)])



train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

val_loader = DataLoader(valset, batch_size=64, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
def imshow(img):

    inverse_transform = transforms.Compose([

      transforms.Normalize(mean = [ 0., 0., 0. ],

                           std = [ 1/0.229, 1/0.224, 1/0.225 ]),

      transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],

                           std = [ 1., 1., 1. ]),

      ])



    unnormalized_img = inverse_transform(img)

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

  



def visualize(): 

    dataiter = iter(train_loader)

    images, labels = dataiter.next()



    classes = dataset.classes

    imshow(torchvision.utils.make_grid(images[:4]))

    for j in range(4):

        print("label: {},  name: {}".format(labels[j].item(),

                                        classes[labels[j]]))

visualize()
def create_model():

    model = models.densenet121(pretrained=True)



    for param in model.parameters():

        param.requires_grad = False



    n_inputs = model.classifier.in_features

    model.classifier = nn.Sequential(

                    nn.Linear(n_inputs, NUM_CLASSES),

                    nn.LogSoftmax(dim=1))

    # Move model to the device specified above



    model.to(device)

    return model
model = create_model()
def get_count_per_class():

    train_labels_count = [0]*NUM_CLASSES

    for ind, dir in enumerate(os.listdir(TRAIN_DIR)):

        path, dirs, files = next(os.walk(os.path.join(TRAIN_DIR, dir)))

        file_count = len(files)

        train_labels_count[ind] = file_count

    return train_labels_count



def create_class_weights():

    dataset_size = len(trainset)

    labels_count = get_count_per_class()

    class_weights = [1-(float(labels_count[class_id])/dataset_size) for class_id in range(NUM_CLASSES)]

  

    return class_weights
weight=torch.FloatTensor(create_class_weights()).to(device)

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
dataset_sizes = {'train': len(trainset), 'valid': len(valset), 'test': len(test_dataset)}
def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=5):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    losses = {'train': [], 'valid':[]}

    acc = {'train': [], 'valid': []}

  

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        print('-' * 10)



    # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:

            if phase == 'train':

                scheduler.step()

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0





            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



            # forward

            # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                    running_loss += loss.item() * inputs.size(0)

                    running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append(epoch_loss)

            acc[phase].append(epoch_acc)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                      phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'valid' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

      time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
def save(model, path):

    torch.save(model, path)



def load(path):

    return torch.load(path)
model = load('../input/model-saved/car_classification_densenet_17e.pt')

dataloaders = {'train': train_loader, 'valid': val_loader}

# model = train_model(dataloaders, model, criterion, optimizer, exp_lr_scheduler, num_epochs=5)

# save(model, "./car_classification_densenet_22e.pt")
confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES)



with torch.no_grad():

    accuracy = 0

    pred_labels = []

    pred_img_ids = []

    true_labels = []

    for i, (inputs, labels) in enumerate(test_loader):

        inputs = inputs.to(device)

        labels = labels.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        running_acc = torch.sum(preds == labels.data)

        accuracy += running_acc

        pred_labels.append(preds)

        true_labels.append(labels)

        

        for t, p in zip(labels.view(-1), preds.view(-1)):

            confusion_matrix[t.long(), p.long()] += 1

        

    for dir in os.listdir(TEST_DIR):

        for file in os.listdir(os.path.join(TEST_DIR, dir)):

            img_id = os.path.splitext(file)[0]

            pred_img_ids.append(img_id)

            

    print('Accuracy =====>>', accuracy.item()/len(test_dataset))
sns.set(font_scale=1.4)#for label size

sns.heatmap(confusion_matrix[:10, :10], annot=True,annot_kws={"size": 16})# font size
pred_labels_expanded = []

for l in pred_labels:

    for l1 in l:

        pred_labels_expanded.append(l1.item())



true_labels_expanded = []

for l in true_labels:

    for l1 in l:

        true_labels_expanded.append(l1.item())



from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelBinarizer



def multiclass_roc_auc_score(truth, pred, average="macro"):



    lb = LabelBinarizer()

    lb.fit(truth)



    truth = lb.transform(truth)

    pred = lb.transform(pred)



    return roc_auc_score(truth, pred, average=average)



print("ROC AUC SCORE: =========>", multiclass_roc_auc_score(true_labels_expanded, pred_labels_expanded))
import pandas as pd

sub_sample = pd.read_csv('../input/virtual-hack/sampleSubmission.csv')

sub_sample.head()
id_label_dict = {}

for id, label in zip(pred_img_ids, pred_labels_expanded):

    id_label_dict[id] = label
od = OrderedDict(sorted(id_label_dict.items()))
my_submission = pd.DataFrame()
my_submission['Id'] = od.keys()

my_submission['Predicted'] = od.values()
my_submission.head()
my_submission.to_csv('my_submission.csv', index=False)