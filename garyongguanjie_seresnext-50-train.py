import pandas as pd

import numpy as np

import cv2

from tqdm import tqdm

import os



DIR = "../input/bengaliai-cv19/"

files = ["train_image_data_0.parquet","train_image_data_1.parquet","train_image_data_2.parquet","train_image_data_3.parquet"]



if not os.path.isfile('train_full_128.npy'):

    all_image_list = []

    for f in files:

        path = DIR + f

        df = pd.read_parquet(path)

        values = 255 - df.iloc[:, 1:].values.reshape(-1, 137, 236).astype(np.uint8)

        img_list = []

        for i in tqdm(range(len(values))):

            img = cv2.resize(values[i],(112,64))

            img_list.append(img)



        img_list = np.array(img_list)

        all_image_list.append(img_list)



    all_image_list = np.concatenate(all_image_list)

    np.save("train_full_128",all_image_list)
import gc

del all_image_list

del img_list

del df

del img

del values

gc.collect()
from torch.utils.data import Dataset

import pandas as pd

import torch

import numpy as np

class BengaliDataset2(Dataset):

    def __init__(self,npy_file,label_csv,aug=None,norm=None):

        self.npy_file = np.load(npy_file)

        self.norm = norm

        df = pd.read_csv(label_csv)

        # for faster access i think

        self.grapheme_root = df["grapheme_root"].values

        self.vowel_diacritic = df["vowel_diacritic"].values

        self.consonant_diacritic = df["consonant_diacritic"].values



        self.aug = aug



    def __getitem__(self, index):

        image_arr = self.npy_file[index]

        # only do this on training

        #use albumentations library

        if self.aug != None:

            image_arr = self.aug(image=image_arr)["image"]



        image_arr = (image_arr/255).astype(np.float32)

        image_arr = torch.from_numpy(image_arr)



        if self.norm != None:

            mean = self.norm['mean']

            std = self.norm['std']

            image_arr = (image_arr -  mean)/std



        grapheme_root = torch.Tensor([self.grapheme_root[index]]).long()

        vowel_diacritic = torch.Tensor([self.vowel_diacritic[index]]).long()

        consonant_diacritic = torch.Tensor([self.consonant_diacritic[index]]).long()

        

        return {"image":image_arr.unsqueeze(0).repeat(3, 1, 1),"grapheme_root":grapheme_root,"vowel_diacritic":vowel_diacritic,"consonant_diacritic":consonant_diacritic}



    def __len__(self):

        return self.npy_file.shape[0]
import albumentations



mean = 13.4/255

std = 40.8/255



from albumentations.core.transforms_interface import ImageOnlyTransform

from typing import Tuple, List, Dict

class ImageTransformer:

    """

    DataAugmentor for Image Classification.

    Args:

        data_augmentations: List of tuple(method: str, params :dict), each elems pass to albumentations

    """



    def __init__(self, data_augmentations: List[Tuple[str, Dict]]):

        """Initialize."""

        augmentations_list = [

            self._get_augmentation(aug_name)(**params)

            for aug_name, params in data_augmentations]

        self.data_aug = albumentations.Compose(augmentations_list)

    

    def __call__(self,image):

        return self.data_aug(image=image)

    

    def __call2__(self, pair: Tuple[np.ndarray]) -> Tuple[np.ndarray]:

        """Forward"""

        img_arr, label = pair

        return self.data_aug(image=img_arr)["image"], label



    def _get_augmentation(self, aug_name: str) -> ImageOnlyTransform:

        """Get augmentations from albumentations"""

        if hasattr(albumentations, aug_name):

            return getattr(albumentations, aug_name)

        else:

            return eval(aug_name)

        

class RandomErasing(ImageOnlyTransform):

    """Class of RandomErase for Albumentations."""



    def __init__(

        self, s: Tuple[float]=(0.02, 0.4), r: Tuple[float]=(0.3, 2.7),

        mask_value_min: int=0, mask_value_max: int=255,

        always_apply: bool=False, p: float=1.0

    ) -> None:

        """Initialize."""

        super().__init__(always_apply, p)

        self.s = s

        self.r = r

        self.mask_value_min = mask_value_min

        self.mask_value_max = mask_value_max



    def apply(self, image: np.ndarray, **params):

        """

        Apply transform.

        Note: Input image shape is (Height, Width, Channel).

        """

        image_copy = np.copy(image)



        # # decide mask value randomly

        mask_value = np.random.randint(self.mask_value_min, self.mask_value_max + 1)



        h, w = image.shape

        # # decide num of pixcels for mask.

        mask_area_pixel = np.random.randint(h * w * self.s[0], h * w * self.s[1])



        # # decide aspect ratio for mask.

        mask_aspect_ratio = np.random.rand() * self.r[1] + self.r[0]



        # # decide mask hight and width

        mask_height = int(np.sqrt(mask_area_pixel / mask_aspect_ratio))

        if mask_height > h - 1:

            mask_height = h - 1

        mask_width = int(mask_aspect_ratio * mask_height)

        if mask_width > w - 1:

            mask_width = w - 1



        # # decide position of mask.

        top = np.random.randint(0, h - mask_height)

        left = np.random.randint(0, w - mask_width)

        bottom = top + mask_height

        right = left + mask_width

        image_copy[top:bottom, left:right].fill(mask_value)



        return image_copy

    

augment = ImageTransformer([('RandomErasing',{'p':0.5})])
train_data = BengaliDataset2("train_full_128.npy","../input/bengaliai-cv19/train.csv",aug =augment,norm={'mean':mean,'std':std})
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import torchvision.models as models

import time

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim import lr_scheduler

import copy

import matplotlib.pyplot as plt

from sklearn.metrics import recall_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
def train_model(model, criterion, optimizer, device, dataloaders, scheduler=None, num_epochs=25):

    since = time.time()



    best_recall = 0.0

    

    dataset_sizes = {'train': len(dataloaders['train'].dataset)}



    train_acc_list = []; train_loss_list= []; val_acc_list = []; val_loss_list = []; unseen_acc_list = []; unseen_loss_list = []



    for epoch in range(num_epochs):

        start = time.time()

        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train']:

            

            #used for calculating recall per epoch

            grapheme_output = []

            vowel_output = []

            consonant_output = []

            grapheme_label = []

            vowel_label = []

            consonant_label = []



            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            grapheme_corrects = 0

            vowel_corrects = 0

            consonant_corrects = 0



            # Iterate over data.

            for i,data in enumerate(dataloaders[phase]):



                inputs = data['image']

                grapheme_root_label = data['grapheme_root']

                vowel_diacritic_label = data['vowel_diacritic']

                consonant_diacritic_label = data['consonant_diacritic']

                inputs = inputs.to(device)



                grapheme_root_label =  grapheme_root_label.to(device)

                vowel_diacritic_label = vowel_diacritic_label.to(device)

                consonant_diacritic_label =  consonant_diacritic_label.to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    g,v,c = model(inputs)

                    

                    grapheme_preds = g.argmax(dim=1)

                    vowel_preds = v.argmax(dim=1) 

                    consonant_preds = c.argmax(dim=1)



                    loss = criterion(g,v,c, grapheme_root_label.squeeze(1),vowel_diacritic_label.squeeze(1),consonant_diacritic_label.squeeze(1))



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                  

                # statistics

                running_loss += loss.item() * inputs.size(0)

                #For accuracy

                grapheme_corrects += torch.sum(grapheme_preds == grapheme_root_label.data.squeeze(1))

                vowel_corrects += torch.sum(vowel_preds == vowel_diacritic_label.data.squeeze(1))

                consonant_corrects += torch.sum(consonant_preds== consonant_diacritic_label.data.squeeze(1))



                if phase == 'train':

                  scheduler.step(epoch+i/dataset_sizes['train'])

                

            if phase == 'val' or phase == 'unseen':

              grapheme_final_output = torch.cat(grapheme_output)    

              grapheme_final_label =  torch.cat(grapheme_label)

              

              vowel_final_output = torch.cat(vowel_output)    

              vowel_final_label =  torch.cat(vowel_label)

              

              consonant_final_output = torch.cat(consonant_output)    

              consonant_final_label =  torch.cat(consonant_label)



              grapheme_recall = recall_score(grapheme_final_output,grapheme_final_label,average='macro')

              vowel_recall = recall_score(vowel_final_output,vowel_final_label,average='macro')

              consonant_recall = recall_score(consonant_final_output,consonant_final_label,average='macro')



            epoch_loss = running_loss / dataset_sizes[phase]

            running_corrects = 0.5*grapheme_corrects.double() + 0.25*vowel_corrects.double() + 0.25*consonant_corrects.double()



            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            epoch_acc = running_corrects / dataset_sizes[phase]



            

          

            if phase == "train":

                # Note this are running values (calculated per batch) rather than actual values at the end of each epoch

                # Decreases training time

                # Not accurate especially at first few epochs

                train_acc_list.append(epoch_acc)

                train_loss_list.append(epoch_loss)

          

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            

                

        end = time.time()

        print(f"time per epoch:{end-start}s")



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val recall: {:4f}'.format(best_recall))



    plots = (train_acc_list,train_loss_list,val_acc_list,val_loss_list,unseen_acc_list,unseen_loss_list)



    return model, plots

def loss(grapheme_root_output,vowel_diacritic_output,consonant_diacritic_output,grapheme_root_label,vowel_diacritic_label,consonant_diacritic_label):

    gloss = nn.CrossEntropyLoss()(grapheme_root_output,grapheme_root_label)

    vloss = nn.CrossEntropyLoss()(vowel_diacritic_output,vowel_diacritic_label)

    closs = nn.CrossEntropyLoss()(consonant_diacritic_output,consonant_diacritic_label)



    return 0.5*gloss + 0.25*vloss + 0.25*closs



def evaluate_test(model,criterion,dataloader,device):

    model.eval()

    running_loss = 0.0

    grapheme_corrects = 0.0

    vowel_corrects = 0.0

    consonant_corrects = 0.0

    

    grapheme_output = []

    vowel_output = []

    consonant_output = []



    grapheme_label = []

    vowel_label = []

    consonant_label = []





    for data in dataloader:



        inputs = data['image']



        grapheme_root_label = data['grapheme_root']

        vowel_diacritic_label = data['vowel_diacritic']

        consonant_diacritic_label = data['consonant_diacritic']



        inputs = inputs.to(device)



        grapheme_root_label =  grapheme_root_label.to(device)

        vowel_diacritic_label = vowel_diacritic_label.to(device)

        consonant_diacritic_label =  consonant_diacritic_label.to(device)



        with torch.no_grad():

            g,v,c = model(inputs)



            loss = criterion(g,v,c, grapheme_root_label.squeeze(1),vowel_diacritic_label.squeeze(1),consonant_diacritic_label.squeeze(1))

            grapheme_preds = g.argmax(dim=1)

            vowel_preds = v.argmax(dim=1)

            consonant_preds = c.argmax(dim=1)



        # statistics

        running_loss += loss.item() * inputs.size(0)

        



        grapheme_corrects += torch.sum(grapheme_preds == grapheme_root_label.data.squeeze(1))

        vowel_corrects += torch.sum(vowel_preds == vowel_diacritic_label.data.squeeze(1))

        consonant_corrects += torch.sum(consonant_preds== consonant_diacritic_label.data.squeeze(1))

        



        grapheme_output.append(grapheme_preds.cpu())

        grapheme_label.append(grapheme_root_label.data.squeeze(1).cpu())

        vowel_output.append(vowel_preds.cpu())

        vowel_label.append(vowel_diacritic_label.data.squeeze(1).cpu())

        consonant_output.append(consonant_preds.cpu())

        consonant_label.append(consonant_diacritic_label.data.squeeze(1).cpu())



    grapheme_final_output = torch.cat(grapheme_output)    

    grapheme_final_label =  torch.cat(grapheme_label)

    

    vowel_final_output = torch.cat(vowel_output)    

    vowel_final_label =  torch.cat(vowel_label)

    

    consonant_final_output = torch.cat(consonant_output)    

    consonant_final_label =  torch.cat(consonant_label)

  



    grapheme_recall = recall_score(grapheme_final_output,grapheme_final_label,average='macro')

    vowel_recall = recall_score(vowel_final_output,vowel_final_label,average='macro')

    consonant_recall = recall_score(consonant_final_output,consonant_final_label,average='macro')



    print("grapheme recall:",grapheme_recall)

    print("vowel_recall:",vowel_recall)

    print("consonant_recall:",consonant_recall)



    print("final recall:",0.5*grapheme_recall+0.25*vowel_recall+0.25*consonant_recall)



    # print(classification_report(grapheme_final_label,grapheme_final_output))

    # fig, axs = plt.subplots()

    # fig.set_figheight(15)

    # fig.set_figwidth(15)

    # cm_vowel = confusion_matrix(grapheme_final_label,grapheme_final_output,normalize='true')

    # cm = ConfusionMatrixDisplay(cm_vowel,[x for x in range(168)])

    # cm.plot(ax=axs)





    loss = running_loss / len(dataloader.dataset)



    running_corrects = 0.5*grapheme_corrects.double() + 0.25*vowel_corrects.double() + 0.25*consonant_corrects.double()



    # epoch_acc = running_corrects.double() / dataset_sizes[phase]

    acc = running_corrects / len(dataloader.dataset)



    print('{} Loss: {:.4f} Acc: {:.4f}'.format(

        "Final Test Accuracy", loss, acc))

    return grapheme_final_output,grapheme_final_label,vowel_final_output,vowel_final_label,consonant_final_output,consonant_final_label
def plot_model_metrics(plots,name):

    train_acc_list,train_loss_list,val_acc_list,val_loss_list,test_acc_list,test_loss_list = plots

    plot(train_acc_list,val_acc_list,test_acc_list,"accuracy",name)

    plot(train_loss_list,val_loss_list,test_loss_list,"loss",name)





def plot(train,val,test,metric,name):

    plt.title(name)

    plt.plot(train,label="train {}".format(metric))

    plt.plot(val,label="val {}".format(metric))

    plt.plot(test,label="test {}".format(metric))

    plt.legend(loc="best")

    plt.savefig("{}-{}".format(name,metric))

    plt.close()
import sys

sys.path.insert(0,"../input/pretrainedmodels/pretrainedmodels-0.7.4")

import pretrainedmodels



class SEModule(nn.Module):

    def __init__(self, channels=2048, reduction=16):

        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,

                             padding=0)

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,

                             padding=0)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        module_input = x

        x = self.avg_pool(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        return module_input * x
# Easier to split stuff up and backpropagate

class MyModel(nn.Module):

  def __init__(self,pretrained=True):

    super().__init__()

    if pretrained:

        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained="imagenet")

    else:

        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=None)

    self.model = nn.Sequential(*list(self.model.children())[:-2])

    

    self.se_g = SEModule()

    self.se_v = SEModule()

    self.se_c = SEModule()

    

    self.avg_pool = nn.AdaptiveAvgPool2d(1)

    

    self.fc_g = nn.Sequential(nn.Linear(2048,512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512,168))

    self.fc_v = nn.Sequential(nn.Linear(2048,512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512,11))

    self.fc_c = nn.Sequential(nn.Linear(2048,512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512,7))

    

  def forward(self,x):

    x = self.model(x)

    

    g = self.se_g(x)

    v = self.se_v(x)

    c = self.se_c(x)

    

    g = torch.flatten(self.avg_pool(g),1)

    v = torch.flatten(self.avg_pool(v),1)

    c = torch.flatten(self.avg_pool(c),1)

    

    g = self.fc_g(g)

    v = self.fc_v(v)

    c = self.fc_c(c)

    return g,v,c
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MyModel()

model.to(device)
train_loader = DataLoader(train_data, batch_size=64, num_workers=2,shuffle=True)

dataloaders = {'train': train_loader}
criterion = loss

optimizer = optim.SGD(model.parameters(),lr=1.5e-02, momentum=0.9, weight_decay=1e-04, nesterov=True)

new_scheduler = lr_scheduler.CosineAnnealingLR(optimizer,40)



model,plots = train_model(model, criterion, optimizer,

            device, dataloaders,scheduler=new_scheduler, num_epochs=40)



torch.save(model.state_dict(),"seresnext_50_2.pth")





print("done")