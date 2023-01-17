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
new_spectrograms = np.array(spectrograms)
labels = pd.get_dummies(df["ID1"]).values
# Convert images to z-score values
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
new_spectrograms = ss.fit_transform(new_spectrograms.reshape(-1, 1)).reshape(new_spectrograms.shape)
pd.DataFrame(new_spectrograms[0].flatten()).describe()
plt.imshow((new_spectrograms[0]), cmap = "RdGy")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import time
import copy
# Set the device (GPU / CPU) depending on what is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}
from albumentations.augmentations.transforms import CoarseDropout, RandomCrop
from albumentations import Compose
albu_trans = Compose(
    [CoarseDropout(max_holes=8, min_holes = 3, max_height=129, max_width=30, min_height = 129, min_width = 5, fill_value=0, p=0.9)]
)
# Implmenting the pytorch abstract class of Dataset:
# Go here and scroll down to dataset class section:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html    
    
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
        sample = {'image': spectrograms, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)
            
        return sample["image"], sample["labels"]
# Compose takes a list of functions to be applied to each element of a dataset
# https://pytorch.org/docs/stable/torchvision/transforms.html

data_transform = transforms.Compose([
        ToTensor()
    ])
from sklearn.model_selection import train_test_split
new_spectrograms = new_spectrograms.astype(np.float32)
train_spectrograms, val_spectrograms, train_y, val_y = train_test_split(new_spectrograms, labels, test_size=0.2, random_state=42, shuffle = True, stratify = labels)
batch_size = 16

train_dataset = torch.utils.data.DataLoader(DolphinsDataset(train_spectrograms, train_y, transform=data_transform, mode = "train"), batch_size=batch_size,
                                             shuffle=True, num_workers=2)

val_dataset = torch.utils.data.DataLoader(DolphinsDataset(val_spectrograms, val_y, transform=data_transform, mode = "val"), batch_size=batch_size,
                                             shuffle=True, num_workers=2)

dataloaders = {"train": train_dataset,
               "val": val_dataset}
from torchvision.models import resnet18

model_ft = resnet18(pretrained="imagenet")
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, labels.shape[1])
model_ft = model_ft.to(device)
criterion = nn.BCEWithLogitsLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
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
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=30)
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
