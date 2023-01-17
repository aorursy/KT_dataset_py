import pandas as pd

import librosa

import matplotlib.pylab as plt

import numpy as np

import random

from scipy import signal

import pickle

%matplotlib inline
df = pd.read_csv("../input/dolphin-vocalizations/signature_whistle_metadata.csv")

signature_whistles = pickle.load(open("../input/dolphin-vocalizations/signature_whistles.p", "rb"))
df = df.loc[df.ID1 == df.ID2]

df = df.drop(166) # This one is wonky

signature_whistles = [signature_whistles[i] for i in df.index]

df.reset_index(inplace=True)
# Get list of dolphin names

dolphins = df.ID1.unique()
# Create spectrograms

sampling_rate = 48000

spectrograms = np.zeros(shape = (len(df), 129, 500))



for i, whistle in enumerate(signature_whistles):

    spectro = signal.spectrogram(whistle, sampling_rate)[-1][:, :500]

    spectrograms[i, :, :spectro.shape[-1]] = spectro
plt.imshow(spectro)
# def plot_sig_whistles_spectogram(dolphin, window_size=None):

    

#     n = 21



#     random.seed(1)

#     exemplars = [spectrograms[i] for i in random.choices(df.loc[df.ID1 == dolphin].index, k=n)]

   

#     fig, axes = plt.subplots(7,3, figsize=(15,15))

#     for i in range(7):

#         for j in range(3):

#             f, t, sxx = exemplars[i*3+j]

#             axes[i,j].pcolormesh(t, f, sxx)

#     fig.suptitle('Signature Whistle Spectrogram for dolphin ' + dolphin, fontsize=20)

#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# for dolphin in dolphins: plot_sig_whistles_spectogram(dolphin)
# def plot_sig_whistles_waveform(dolphin, same_scale=True):

    

#     n = 21



#     random.seed(1)

#     idx = random.choices(df.loc[df.ID1 == dolphin].index, k=n)

#     sigs = [signature_whistles[i] for i in idx]

    

#     max_val = 0

#     min_val = 0

#     for i in range(n):

#         # print(sigs[i])

#         val = np.max(sigs[i])

#         # print(val)

#         if val > max_val: max_val = val

#         val = np.min(sigs[i])

#         if val < min_val: min_val = val



#     fig, axes = plt.subplots(7,3, figsize=(15,15))  

#     for i in range(7):

#         for j in range(3):

#             axes[i,j].plot(sigs[i*3+j])

#             if same_scale: axes[i,j].set_ylim([min_val,max_val])

#             lbl = "D:" + str(df.loc[idx[i*3+j],'day']) + "; Sess:" + str(df.loc[idx[i*3+j],'session']) + "; Ch:" + str(df.loc[idx[i*3+j],'channel'])

#             axes[i,j].set_title(lbl + "; Start: " + df.loc[idx[i*3+j],'start_time'] + "; End: " + df.loc[idx[i*3+j],'end_time'])

#     fig.suptitle('Signature Whistle Waveforms for dolphin ' + dolphin, fontsize=20)

#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# for dolphin in dolphins: plot_sig_whistles_waveform(dolphin, same_scale=True)
import cv2

# new_spectrograms = []

# sizes = []



# for spectrogram in spectrograms:

#     new_spectrograms.append(np.log1p(cv2.resize(spectrogram[-1], (500, 129))))

#     sizes.append(spectrogram[-1].shape)

# pd.DataFrame(np.array(sizes)[:, 1]).hist(bins = 10)
new_spectrograms = np.array(spectrograms)

labels = pd.get_dummies(df["ID1"]).values
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

new_spectrograms = ss.fit_transform(new_spectrograms.reshape(-1, 1)).reshape(new_spectrograms.shape)
pd.DataFrame(new_spectrograms[0].flatten()).describe()
plt.imshow((new_spectrograms[3]), cmap = "RdGy")
import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

from torch.utils.data import Dataset, DataLoader

import time

import copy
class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, labels = sample['image'], sample['labels']

#         print(type(image), type(labels))

        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),

                'labels': torch.from_numpy(labels)}
from albumentations.augmentations.transforms import CoarseDropout, RandomCrop

from albumentations import Compose

albu_trans = Compose(

    [

        CoarseDropout(max_holes=8, min_holes = 3, max_height=129, max_width=30, min_height = 129, min_width = 5, fill_value=0, p=0.9),

#         RandomCrop(height = 129, width = 50)

    ]

)

        

class DolphinsDataset(Dataset):

    def __init__(self, spectrograms, labels, transform=None, mode = "train"):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.spectrograms = spectrograms

        self.labels = labels

        self.transform = transform

        self.mode = mode

        



    def __len__(self):

        return len(self.spectrograms)



    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

        labels = self.labels[idx].astype(np.float64)

        spectrograms = self.spectrograms[idx].astype(np.float32)

        spectrograms = np.repeat(spectrograms[:, :, None], axis = 2, repeats=3)

        

        if self.mode == "train":

            

            spectrograms = albu_trans(image = spectrograms)["image"]

#             print("train mode")

#             plt.imshow(spectrograms)

#             plt.show()

#         print(spectrograms[:, :, None].shape,

#              labels.shape)

        sample = {'image': spectrograms, 'labels': labels}

        if self.transform:

            sample = self.transform(sample)

            

        return sample["image"], sample["labels"]
data_transform = transforms.Compose([

        ToTensor()

    ])
new_spectrograms = new_spectrograms.astype(np.float32)
from sklearn.model_selection import train_test_split

train_spectrograms, val_spectrograms, train_y, val_y = train_test_split(new_spectrograms, labels, test_size=0.2, random_state=42, shuffle = True, stratify = labels)
batch_size = 16

train_dataset = torch.utils.data.DataLoader(DolphinsDataset(train_spectrograms, train_y, transform=data_transform, mode = "train"), batch_size=batch_size,

                                             shuffle=True, num_workers=2)

val_dataset = torch.utils.data.DataLoader(DolphinsDataset(val_spectrograms, val_y, transform=data_transform, mode = "val"), batch_size=batch_size,

                                             shuffle=True, num_workers=2)


dataloaders = {"train": train_dataset,

               "val": val_dataset}
for batch in dataloaders["train"]:

    break
plt.imshow(batch[0].numpy()[7].transpose((1, 2, 0)))
for batch in dataloaders["val"]:

#     print(batch)

    break
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 100

    dataset_sizes = {}

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

            running_corrects = 0.0



            # Iterate over data.

            dataset_sizes[phase] = len(dataloaders[phase])

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)

#                 print(labels.dtype)

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



                # statistics

                _, labels = torch.max(labels, 1)

                running_loss += loss.item() * inputs.size(0)

                

                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':

                scheduler.step()



            epoch_loss = running_loss / (dataset_sizes[phase] * batch_size)

            epoch_acc = running_corrects.double() / (dataset_sizes[phase] * batch_size)



            print('{} Loss: {:.4f} Acc: {:.4f} '.format(

                phase, epoch_loss, epoch_acc

            ))



            # deep copy the model

            if phase == 'val' and epoch_loss < best_loss:

                best_loss = epoch_loss

                best_model_wts = copy.deepcopy(model.state_dict())

                best_acc = epoch_acc





    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
from torchvision.models import resnet18
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = resnet18(pretrained="imagenet")

num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.

# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

model_ft.fc = nn.Linear(num_ftrs, labels.shape[1])



model_ft = model_ft.to(device)



criterion = nn.BCEWithLogitsLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=30)
# albu_trans = Compose(

#     [

#         Cutout(num_holes=64, max_h_size=32, max_w_size=32, fill_value=0, p=0.1),

# #         RandomCrop(height = 129, width = 50)

#     ]

# )

# train_dataset = torch.utils.data.DataLoader(DolphinsDataset(train_spectrograms, train_y, transform=data_transform, mode = "train"), batch_size=batch_size,

#                                              shuffle=True, num_workers=2)

# val_dataset = torch.utils.data.DataLoader(DolphinsDataset(val_spectrograms, val_y, transform=data_transform, mode = "val"), batch_size=batch_size,

#                                              shuffle=True, num_workers=2)

# dataloaders = {"train": train_dataset,

#                "val": val_dataset}
# # Observe that all parameters are being optimized

# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)



# # Decay LR by a factor of 0.1 every 7 epochs

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

#                        num_epochs=30)
from sklearn.metrics import accuracy_score

preds = []

val_labels = []

model_ft = model_ft.eval()

for i, (specto, label) in enumerate(zip(np.repeat(val_spectrograms[:, None, :, :], axis = 1, repeats=3), val_y)):

    specto = torch.Tensor(specto[None, :, :, :]).to(device)

    preds.append(model_ft(specto).detach().cpu().numpy().argmax())

    val_labels.append(label.argmax())

    if preds[-1] != val_labels[-1]:

        print(preds[-1], val_labels[-1])

        plt.imshow(np.log1p(val_spectrograms[i]), cmap = "RdGy")

        plt.show()

print(accuracy_score(preds, val_labels))
from sklearn.metrics import confusion_matrix
confusion_matrix(val_labels, preds)