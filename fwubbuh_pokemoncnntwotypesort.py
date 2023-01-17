from __future__ import print_function, division



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

from sklearn.model_selection import train_test_split

from PIL import Image

import csv

import warnings  

warnings.filterwarnings('ignore')
def column(matrix, i):

    return [row[i] for row in matrix]



class PokemonDataset(object):

    def __init__(self, root, transforms):

        self.root = root

        self.transforms = transforms

        # load all image files, sorting them to

        # ensure that they are aligned

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images", "images"))))

        with open(os.path.join(root, "pokemon.csv"), newline='') as f:

            reader = csv.reader(f)

            data = list(reader)

        self.data = data

        self.All_names = column(data, 0)

        self.classes = ['Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel',

               'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark', 'Fairy', '']



    def __getitem__(self, idx):

        # load images ad masks

        img_path = os.path.join(self.root, "images", "images", self.imgs[idx])

        img = Image.open(img_path).convert("RGB")



        image_id = torch.tensor([idx])

        image_name = str(self.imgs[idx])[:-4]

        index = self.All_names.index(image_name)

        tester = self.data[index]



        target = {}

        target["pokedex"] = self.All_names.index(image_name)

        target["image_id"] = image_id

        target["name"] = image_name

        target["label"] = tester[1]

        target["label2"] = ''

        types = torch.tensor([0]*len(self.classes))

        types[self.classes.index(tester[1])] = 1

        if(len(tester) == 3):

            target["label2"] = tester[2]

            types[self.classes.index(tester[2])] = 1

        else:

            target["label2"] = ''

            types[-1] = 1



        if self.transforms is not None:

            img = self.transforms(img)



        return img, types, target



    def __len__(self):

        return len(self.imgs)
data_transforms = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

    

def train_val_dataset(dataset, val_split=0.25):

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)

    datasets = {}

    datasets['train'] = torch.utils.data.Subset(dataset, train_idx)

    datasets['val'] = torch.utils.data.Subset(dataset, val_idx)

    return datasets
data_dir = '/kaggle/input/pokemon-images-and-types/'

x = 'images'

dataset = PokemonDataset(data_dir, transforms=data_transforms)

image_datasets = train_val_dataset(dataset)

classes = ['Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel',

           'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark', 'Fairy', '']

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,

                                             shuffle=True, num_workers=4)

              for x in ['train', 'val']}

    

inputs, types, cat  = next(iter(dataloaders['train']))

print(inputs.shape)

sub_names = cat["name"]

sub_types = cat["label"]

sub_types2 = cat["label2"]

# Make a grid from batch

out = torchvision.utils.make_grid(inputs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

    

def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated

 

imshow(out, title=sub_names)
def myCrit(xs, labs):

    return torch.sum(-torch.sum(torch.log(xs)*labs + torch.log(1 - xs)*(1 - labs), dim = 1))# + torch.log(torch.sum(torch.exp(xs), dim = 1)))



def train_model(model, optimizer, scheduler, num_epochs=25):

    since = time.time()

    

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

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

            for inputs, labels, _ in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)

   

                # zero the parameter gradients

                optimizer.zero_grad()

    

                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = torch.sigmoid(model(inputs))

                    outputs_2 = outputs.clone()

                    preds = outputs.argmax(1)

                    preds_0 = torch.zeros(outputs.shape).to(device).scatter(1, preds.unsqueeze(1), 1)

                    outputs_2 = outputs_2*(1 - preds_0)

                    preds2 = outputs_2.argmax(1)

                    preds_f = preds_0 + torch.zeros(outputs.shape).to(device).scatter(1, preds2.unsqueeze(1), 1)

                    loss = myCrit(outputs, labels)

    

                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

    

                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += .5*torch.sum(labels.data*preds_f)

            if phase == 'train':

                scheduler.step()

    

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

    

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))

    

            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

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
def visualize_model(model, num_images=10):

    was_training = model.training

    model.eval()

    images_so_far = 0

    fig = plt.figure()

    

    with torch.no_grad():

        for i, (inputs, labels, _) in enumerate(dataloaders['val']):

            inputs = inputs.to(device)

            labels = labels.to(device)

    

            outputs = torch.sigmoid(model(inputs))

            outputs = outputs.to(device)

            outputs_2 = outputs.clone()

            preds = outputs.argmax(1)

            preds = preds.to(device)

            preds_0 = torch.zeros(outputs.shape).to(device).scatter(1, preds.unsqueeze(1), 1)

            outputs_2 = outputs_2*(1 - preds_0)

            preds2 = outputs_2.argmax(1)

    

            for j in range(inputs.size()[0]):

                images_so_far += 1

                ax = plt.subplot(num_images//2, 2, images_so_far)

                ax.axis('off')

                ax.set_title('predicted: {}'.format(classes[preds[j]] + ',' + classes[preds2[j]]))

                imshow(inputs.cpu().data[j])

    

                if images_so_far == num_images:

                    model.train(mode=was_training)

                    return

        model.train(mode=was_training)
model_conv = torchvision.models.resnet50(pretrained=True)

for param in model_conv.parameters():

    #Change this to false to train over just the output layer, easier if no GPU available

    param.requires_grad = True

num_ftrs = model_conv.fc.in_features

model_conv.fc = nn.Linear(num_ftrs, len(classes))

    

model_conv = model_conv.to(device)

    

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9, weight_decay=1)

    

# Decay LR by a factor of 0.15 every 4 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=4, gamma=0.15)

model_conv = train_model(model_conv, optimizer_conv, exp_lr_scheduler, num_epochs=25)

visualize_model(model_conv)



plt.ioff()

plt.show()
import scipy.signal as signal

import statistics



dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,

                                                 shuffle=False, num_workers=0)

model = model_conv

with torch.no_grad():

    indexes = []

    names = []

    corr = []

    type_count = torch.tensor([0]*len(classes))

    type_corr = torch.tensor([0]*len(classes))

    counter = 1

    for i, (inputs, labels, info) in enumerate(dataloader):

        inputs = inputs.to(device)

        labels = labels.to(device)

 

        outputs = torch.sigmoid(model(inputs))

        outputs_2 = outputs.clone()

        a = outputs.argmax(1)

        preds_0 = torch.zeros(outputs.shape).to(device).scatter(1, a.unsqueeze(1), 1)

        outputs_2 = outputs_2*(1 - preds_0)

        b = outputs_2.argmax(1)

        preds_1 = torch.zeros(outputs.shape).to(device).scatter(1, b.unsqueeze(1), 1)

        preds = preds_0 + preds_1

        corrs = .5*torch.sum(labels*preds, dim = 1)

        

        corrs = corrs.to('cpu')

        labels = labels.to('cpu')

        for j in range(inputs.size()[0]):

            indexes.append(info["pokedex"][j].tolist())

            names.append(info["name"][j])

            type_count = type_count + labels[j]

            type_corr = type_corr + corrs[j]*labels[j]

            corr.append(.5*torch.sum(corrs[j]*labels[j]))

            counter += 1

        

    type_avg = (type_corr/type_count)

    type_count = type_count.tolist()

    type_corr = type_corr.tolist()

    type_avg = type_avg.tolist()

    

corr = [correct.tolist() for (ind,correct) in sorted(zip(indexes, corr))]

names = [name for (ind,name) in sorted(zip(indexes, names))]

indexes = sorted(indexes)

    

smoothCorr = signal.savgol_filter(corr, 25, 1)

#plt.plot(indexes, corr, 'go')

plt.plot(indexes, smoothCorr)

plt.xlabel("Pokedex Number")

plt.ylabel("p correct")

plt.show()

    

pertype = type_count/(np.sum(type_count))

sortClass = [classy for (per, classy) in sorted(zip(type_avg, classes))]

sortPer = [tp for (per, tp) in sorted(zip(type_avg, pertype))]

x_pos = [i for i, _ in enumerate(classes)] 



plt.bar(x_pos, sorted(type_avg), color = 'green')

plt.xlabel("Pokemon Type")

plt.ylabel("p correct")     

plt.xticks(x_pos, sortClass, rotation = 90)

plt.show()



plt.bar(x_pos, sorted(type_avg), color = 'green')

plt.bar(x_pos, sortPer, color='red')

plt.xlabel("Pokemon Type")

plt.ylabel("p correct and p of type")    

plt.xticks(x_pos, sortClass, rotation = 90)

plt.show()

    

plt.scatter(sortPer[:-1], sorted(type_avg)[:-1])

plt.xlabel('p of Pokemon of Type')

plt.ylabel('p correct')

plt.show()

    

gens = ['gen 1', 'gen 2', 'gen 3', 'gen 4', 'gen 5', 'gen 6', 'gen 7']

pokenum = [151, 251, 386, 493, 649, 721, 809]

corr_by_gen = [0]*len(gens)

corr_by_gen[0] = statistics.mean(corr[0:pokenum[0]])

for i in range(1, len(pokenum)):

    corr_by_gen[i] = statistics.mean(corr[pokenum[i-1]:pokenum[i]])

 

gen_pos = [i for i, _ in enumerate(gens)]    

plt.bar(gen_pos, corr_by_gen, color='green')

plt.xlabel("Generation Number")

plt.ylabel("p correct")

    

plt.xticks(gen_pos, gens)

   

plt.show()
