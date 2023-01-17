# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings 

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import cv2

import os.path as osp

import time

import torch

import torch.nn as nn

import torch.optim as optim 

from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook as tqdm

from PIL import Image



import matplotlib.pyplot as plt



# now we train a binary classifier to check whether one image is defective or not 

TRAINVAL_ANNOT = "../input/severstal-steel-defect-detection/train.csv"

TRAINVAL_IMAGE_ROOT = "../input/severstal-steel-defect-detection/train_images/"



# Any results you write to the current directory are saved as output.
def get_annot(annot_path):

    trainval_annot = pd.read_csv(TRAINVAL_ANNOT)

    trainval_annot['ImageId'] = trainval_annot['ImageId_ClassId'].apply(lambda x: x.split("_")[0])

    trainval_annot['ClassId'] = trainval_annot['ImageId_ClassId'].apply(lambda x: x.split("_")[1])

    trainval_annot['HasMask'] = trainval_annot['EncodedPixels'].notnull().astype('int')



    trainval_annot = trainval_annot.groupby('ImageId').agg(np.sum).sort_values(by="HasMask", ascending=False).reset_index()

    trainval_annot['AllMissing'] = trainval_annot['HasMask'] == 0

    trainval_annot = trainval_annot.drop('HasMask', axis=1)



    train_annot, val_annot = train_test_split(trainval_annot, test_size=0.15)

    print("{}/{} images for train/val.".format(len(train_annot), len(val_annot)))

    train_defect_num = (train_annot['AllMissing'] == False).sum()

    val_defect_num = (val_annot['AllMissing'] == False).sum()

    print("{}/{} defective images in train/val set.".format(train_defect_num, val_defect_num))

    return {"train": train_annot, "val": val_annot}
def get_transform(phase):

    list_transform = []

    if phase == 'train':

        list_transform.extend([

            # transforms.RandomResizedCrop(224),

            transforms.RandomHorizontalFlip(),

            transforms.RandomVerticalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

    else:

        list_transform.extend([

            # transforms.Resize(256),

            # transforms.CenterCrop(224),

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

    return transforms.Compose(list_transform)





class SteelDataset(Dataset):

    def __init__(self, annot, image_folder, phase):

        self.annot = annot

        self.image_folder = image_folder

        self.phase = phase

        self.transform = get_transform(phase)

        

    def __getitem__(self, index):

        row = self.annot.iloc[index, :]

        image_path = osp.join(self.image_folder, row['ImageId'])

        target = int(row['AllMissing'])

        image = Image.open(image_path)

        image = self.transform(image)

        return image, target

    

    def __len__(self):

        return len(self.annot)

    

    

def get_dataloader(annot, image_folder, phase, batch_size=16, num_workers=4):

    dataset = SteelDataset(annot, image_folder, phase)

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)



def get_model(model_name):

    # for convenience, only add resnet

    model = models.__dict__[model_name]()

    if model_name.startswith("resnet"):

        in_features = model.fc.in_features

        model.fc = nn.Linear(in_features, 1)

    else:

        raise KeyError("Only support resnet!")

    return model
class Trainer:

    def __init__(self,  model_name="resnet34", pretrained=False):

        self.lr = 5e-4

        self.threshold = 0.5

        self.best_acc = 0.0

        self.device = torch.device("cuda:0")

        self.model_path = "./binary_classification_model.pth"

        self.pretrained = pretrained

        self.pretrained_model_path = osp.join("../input/severstal-binary-classifier/", self.model_path)

        self.num_epochs = 35

        self.annot_path = TRAINVAL_ANNOT

        self.image_folder = TRAINVAL_IMAGE_ROOT

        self.phases = ['train', 'val']

        self.batch_sizes = {'train': 16, 'val': 64}

        self.model = get_model(model_name).to(self.device)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        self.criterion = nn.functional.binary_cross_entropy_with_logits

        self.annots = get_annot(self.annot_path)

        self.dataloaders = {phase: get_dataloader(self.annots[phase], self.image_folder, 

                                phase, self.batch_sizes[phase]) for phase in self.phases}

        self.losses = {phase: [] for phase in self.phases}

        self.accuracies = {phase: [] for phase in self.phases}

        

    def forward(self, inputs, targets):   

        inputs = inputs.to(self.device)

        targets = targets.to(self.device)

        outputs = self.model(inputs)

        targets = targets.unsqueeze(1).float()

        loss = self.criterion(outputs, targets)

        return loss, outputs

        

    def iterate(self, epoch, phase):

        start = time.time()

        print("Epoch: {} | Phase: {}".format(epoch, phase))

        self.model.train(phase == 'train')

        dataloader = self.dataloaders[phase]

        running_loss = 0.0

        running_corrects = 0

        self.optimizer.zero_grad()

        tk = tqdm(dataloader, total=len(dataloader))

        torch.set_grad_enabled(phase == 'train')

        for idx, batch in enumerate(tk):

            inputs, targets = batch

            loss, outputs = self.forward(inputs, targets)

            if phase == 'train':

                loss.backward()

                self.optimizer.step()

                self.optimizer.zero_grad()

            preds = (torch.sigmoid(outputs) > self.threshold).reshape_as(targets).long().cpu()

            running_corrects += (preds == targets).sum().item()

            running_loss += loss.item() * inputs.size(0)

            tk.set_postfix(loss=running_loss / ((idx+1) * self.batch_sizes[phase]))

            tk.update()

        torch.set_grad_enabled(phase == 'train')

        

        dataset_size = len(dataloader.dataset)

        running_loss /= dataset_size

        running_acc = running_corrects / dataset_size

        self.losses[phase].append(running_loss)

        self.accuracies[phase].append(running_acc)

        end  = time.time()

        time_elapsed = int(end - start)

        print("Finished in {} mins and {} secs, loss: {:.3f}, acc: {:.3f}%".format(

            time_elapsed // 60, time_elapsed % 60, running_loss, running_acc * 100))

        return running_acc

        

    def save_model(self, epoch):

        state = {

            "epoch": epoch,

            "best_acc": self.best_acc, 

            "state_dict": self.model.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict()

        }

        print("****** Find new optimal model, saving to disk ******")

        torch.save(state, self.model_path)

        return 

    

    def summary(self):

        print("Training finished. Best val acc: {:.3f}%".format(self.best_acc))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        len_x = len(self.losses['train'])

        ax1.plot(range(len_x), self.losses['train'], label="train loss")

        ax1.plot(range(len_x), self.losses['val'], label="val loss")

        ax1.title("loss curve");   ax1.xlabel("epoch");   ax1.ylabel("loss")

        ax2.plot(range(len_x), self.accuracies['train'], label='train acc')

        ax2.plot(range(len_x), self.accuracies['val'], label='val acc')

        ax2.title("accuracy curve");  ax2.xlabel("epoch");  ax2.ylabel("accuracy")

        plt.show()

    

    def start(self):

        resume_epoch = 0

        if self.pretrained or osp.exists(self.model_path):

            state = torch.load(self.pretrained_model_path) if self.pretrained else torch.load(self.model_path)

            resume_epoch = state['epoch'] + 1

            self.best_acc = state['best_acc']

            self.optimizer.load_state_dict(state['optimizer_state_dict'])

            self.model.load_state_dict(state['state_dict'])

            print("Load checkpoint from {}, resume training from {} epoch with best acc {}%".format(

                    self.model_path, resume_epoch, self.best_acc * 100))

            

        for epoch in range(resume_epoch, self.num_epochs):

            train_acc = self.iterate(epoch, 'train')

            self.scheduler.step()

            val_acc = self.iterate(epoch, 'val')

            if val_acc > self.best_acc:

                self.best_acc = val_acc

                self.save_model(epoch)

        self.summary()
trainer = Trainer("resnet34", pretrained=False)

trainer.start()
!nvidia-smi