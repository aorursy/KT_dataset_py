import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from torchvision import datasets, models, transforms

import torch.nn as nn

import torch.optim as optim

import torch

from torch.utils.data import DataLoader, Dataset

from torchvision.utils import make_grid

from torch.optim import lr_scheduler

import matplotlib.pyplot as plt

from collections import OrderedDict

import os

from functools import reduce

import random

import skimage.io

import importlib

from PIL import Image

#import pretrainedmodels

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score

from itertools import product

import copy
model_name="wide_resnet50_2"

initial_lr=0.01

l2_=0.0005

cycles=6

image_shape=224

epoch_cycle=60

n_predictions=4

weight_file = "ens_%s_%d_l2_%.6f.pt" % (model_name,image_shape,l2_)



is_submitting=os.path.exists('../input/prostate-cancer-grade-assessment/test_images')

training_mode = False

on_kaggle=True





filepath = "../input/prostate-cancer-grade-assessment/train_images/fdb9d38ce2b5a1d31bb030cd2d3a03b9.tiff"

train_csv = "../input/prostate-cancer-grade-assessment/train.csv"

test_csv = "../input/prostate-cancer-grade-assessment/test.csv"

train_path = "../input/prostate-cancer-grade-assessment/train_images"

submission_path = '../input/prostate-cancer-grade-assessment/test_images'



sub_data = pd.read_csv(test_csv)

full = pd.read_csv(train_csv)

#Subset train.csv to rows for which images exist in "train_images" folder. Useful for test environment that dont contain all images

full_files = os.listdir("../input/prostate-cancer-grade-assessment/train_images")

full_files = [f.replace(".tiff", "") for f in full_files]

available_files = list(set(full_files) & set(full['image_id']))

full = full.loc[full['image_id'].isin(available_files), :]



train, test = train_test_split(full, test_size=.2,random_state=123)
data_transform = {

    "train": transforms.Compose([

        transforms.RandomResizedCrop(image_shape,scale=(0.5,0.8)),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])

    ]),

    "test": transforms.Compose([

        transforms.RandomResizedCrop(image_shape,scale=(0.5,0.8)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])

    ])

}






def tile(image, size=50):

    h, w, c = image.shape

    h_pad = size - h % size

    w_pad = size - w % size

    image2 = np.pad(image, ((0, h_pad), (0, w_pad), (0, 0)), constant_values=255)



    n_h = int(image2.shape[0] / size)

    n_w = int(image2.shape[1] / size)



    tiles = list(product(range(n_h), range(n_w)))

    #drop tiles that are blank

    tiles2=[]

    for t in tiles:

        t_h, t_w = t

        section=image2[t_h * size:(t_h + 1) * size, t_w * size:(t_w + 1) * size, :]

        if not np.all(section.reshape((-1,))==255):

            tiles2.append(section)

    if len(tiles2)==0:

        print("Warning: blank image: returning white image of shape 500,500,3")

        return np.zeros((500, 500, 3), dtype='uint8') + 255



    n_total = len(tiles2)

    sh = int(np.ceil(np.sqrt(n_total)))



    im = np.zeros((sh * size, sh * size, 3), dtype='uint8') + 255



    random.shuffle(tiles2)



    k = 0

    for i in range(sh):

        for j in range(sh):

            if k == len(tiles2): break

            im[i * size:(i + 1) * size, j * size:(j + 1) * size, :] = tiles2[k]

            k = k + 1

    return im





def load_image(filepath):

    img = skimage.io.MultiImage(filepath)

    image = img[-1]

    image = tile(image)

    image = Image.fromarray(image).convert("RGB")

    return image



test_image=load_image(filepath)

plt.imshow(test_image)

plt.show()
class PandaDataset(Dataset):

    def __init__(self, df, image_dir, labels=False, transforms=None):

        self.df = df

        self.transforms = transforms

        self.image_dir = image_dir

        self.labels = labels # if labels is set to false, the __getitem__ function returns 0 as the label



    def __len__(self):

        return len(self.df)



    def __getitem__(self, item):

        image = load_image(os.path.join(self.image_dir, self.df['image_id'].iloc[item] + ".tiff"))

        if self.transforms:

            image = self.transforms(image)

        if self.labels:

            labs = self.df['isup_grade'].iloc[item]

            return image, labs

        else:

            return image,0

train_df = PandaDataset(df=train, labels=True, image_dir=train_path, transforms=data_transform['train'])

trainloader = DataLoader(train_df, batch_size=12,

                         shuffle=True, num_workers=4)



test_df = PandaDataset(df=test, labels=True, image_dir=train_path, transforms=data_transform['test'])

testloader = DataLoader(test_df, batch_size=12, shuffle=False, num_workers=4)



submission_df = PandaDataset(sub_data, image_dir=submission_path, labels=False, transforms=data_transform['test'])

submission_loader = DataLoader(submission_df, batch_size=12, shuffle=False, num_workers=4)
data_iter=iter(trainloader)

ss,_=data_iter.next()

grid=make_grid(ss, nrow=3)

plt.imshow(grid.permute(1,2,0))

plt.show()
def schedule_fun(t,lr_max=initial_lr,lr_min=0.0,T=epoch_cycle):

    t=t%T

    return lr_min+0.5*(lr_max-lr_min)*(1+np.cos(t/T*np.pi))



learning_rate=[schedule_fun(i,lr_max=0.1,lr_min=0.0, T=60) for i in range(300)]

plt.plot(learning_rate)

plt.xlabel("Epoch")

plt.ylabel("Learning rate")

plt.show()
model=getattr(models,model_name)(pretrained=training_mode)

#model = model = pretrainedmodels.__dict__[model_name](num_classes=1000,pretrained='imagenet')

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 6)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)



optimizer = optim.SGD(model.parameters(), lr=1.0)

scheduler = lr_scheduler.LambdaLR(optimizer,schedule_fun,last_epoch=-1)

criterion = nn.CrossEntropyLoss()
def kappa(y1, y2):

    return cohen_kappa_score(y1, y2, labels=[0, 1, 2, 3, 4, 5], weights='quadratic')
weight_list = []



epochs = cycles*epoch_cycle

epoch = 0

if training_mode:

    losses=[]

    kappas=[]

    for epoch in range(epoch, epochs):

        print("Epoch %d: learning rate=%.6f" % (epoch, scheduler.get_last_lr()[0]))

        train_loss = 0.0

        test_loss = 0.0

        kappa_train = []

        kappa_test = []

        model.train()

        for images, labels in trainloader:

            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)



            loss.backward()



            optimizer.step()

            pred_train = logits.argmax(dim=1)

            kappa_train.append(kappa(pred_train.cpu(), labels.cpu()))

            train_loss += loss.item() * images.size(0)



        model.eval()

        for images, labels in testloader:

            with torch.no_grad():

                images = images.to(device)

                labels = labels.to(device)

                logits = model(images)

                loss = criterion(logits, labels)

                pred_test = logits.argmax(dim=1)

                kappa_test.append(kappa(pred_test.cpu(), labels.cpu()))

                test_loss += loss.item() * images.size(0)

        print("Epoch %d: loss %.3f val_loss %.3f | kappa %.3f val_kappa %.3f" % (

            epoch, train_loss / train_df.__len__(), test_loss / test_df.__len__(), np.mean(kappa_train),

            np.mean(kappa_test)))

        losses.append((train_loss / train_df.__len__(), test_loss / test_df.__len__()))

        kappas.append((np.mean(kappa_train), np.mean(kappa_test)))

        if (epoch+1) % epoch_cycle==0:

            weight_list.append(model.state_dict())

        scheduler.step()



    torch.save(weight_list,weight_file)



    #plot of training/test loss and kappa

    fig,axes=plt.subplots(1,2)

    axes[0].plot(losses)

    axes[0].set_ylim((np.min(losses),min(3,np.max(losses))))

    axes[0].set_title("Losses")

    axes[1].plot(kappas)

    axes[1].set_title("Kappa")

    if display:

        plt.show()

    else:

        plt.savefig("/home/pmwaniki/Dropbox/tmp/finetune_ens_%s_%d" % (model_name,image_shape))
if on_kaggle:

    weight_file2=os.path.join("../input/results3",weight_file)

else:

    weight_file2=weight_file









if not training_mode: weight_list=torch.load(weight_file2,map_location=device)



pred_eval_ens=[]

for i in range(len(weight_list)):

    pred_eval=[]

    model.load_state_dict(weight_list[i])

    model.eval()

    for p in range(n_predictions):

        pp=list()

        for images, labels in testloader:

            with torch.no_grad():

                images = images.to(device)

                logits = model(images)

                probs=torch.nn.Softmax(dim=1)(logits)

                if torch.cuda.is_available():

                    probs=probs.cpu()

                pp.append(probs)

        pp=np.concatenate(pp)

        pred_eval.append(pp)

    pred_eval_ens.append(np.stack(pred_eval).mean(axis=0))

pred_eval_ens2=np.stack(pred_eval_ens,axis=0)

pred_eval_ens3=pred_eval_ens2.mean(axis=0)

pred_eval_cat=np.argmax(pred_eval_ens3,axis=1)

print("Kappa ensemble eval= %.3f" % kappa(test['isup_grade'],pred_eval_cat))
if is_submitting:

    pred_test_ens = []

    for i in range(len(weight_list)):

        pred_eval = []

        model.load_state_dict(weight_list[i])

        model.eval()

        for p in range(n_predictions):

            pp = list()

            for images, labels in submission_loader:

                with torch.no_grad():

                    images = images.to(device)

                    logits = model(images)

                    probs = torch.nn.Softmax(dim=1)(logits)

                    if torch.cuda.is_available():

                        probs = probs.cpu()

                    pp.append(probs)

            pp = np.concatenate(pp)

            pred_eval.append(pp)

        pred_test_ens.append(np.stack(pred_eval).mean(axis=0))

    pred_test_ens2 = np.stack(pred_test_ens, axis=0)

    pred_test_ens3 = pred_test_ens2.mean(axis=0)

    pred_test_cat = np.argmax(pred_test_ens3, axis=1)

else:

    rand_preds = []

    for i in range(len(sub_data)):

        rand_preds.append(random.randint(0, 5))

    pred_test_cat=np.array(rand_preds)



sub_data['isup_grade'] = pred_test_cat

test_df = sub_data[["image_id", "isup_grade"]]

test_df.to_csv('submission.csv', index=False)

test_df.head()
