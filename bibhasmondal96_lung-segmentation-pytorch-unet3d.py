%%writefile unet3d.py

import torch

from torch import nn

import torch.nn.functional as F



# Ref https://github.com/Thvnvtos/Lung_Segmentation



# __                            __

#  1|__   ________________   __|1

#     2|__  ____________  __|2

#        3|__  ______  __|3

#           4|__ __ __|4



class ConvUnit(nn.Module):

    """

        Convolution Unit: (Conv3D -> BatchNorm -> ReLu) * 2

    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.double_conv = nn.Sequential(

            nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost



            nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 1),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)

          )



    def forward(self,x):

        return self.double_conv(x)



class EncoderUnit(nn.Module):

    """

    An Encoder Unit with the ConvUnit and MaxPool

    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.MaxPool3d(2),

            ConvUnit(in_channels, out_channels)

        )

    def forward(self, x):

        return self.encoder(x)



class DecoderUnit(nn.Module):

    """

    ConvUnit and upsample with Upsample or convTranspose

    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = ConvUnit(in_channels, out_channels)



    def forward(self, x1, x2):

        x1 = self.up(x1)



        diffZ = x2.size()[2] - x1.size()[2]

        diffY = x2.size()[3] - x1.size()[3]

        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])



        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)



class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)



    def forward(self, x):

        return self.conv(x)



class UNet3d(nn.Module):

    def __init__(self, in_channels, n_classes, s_channels):

        super().__init__()

        self.in_channels = in_channels

        self.n_classes = n_classes

        self.s_channels = s_channels



        self.conv = ConvUnit(in_channels, s_channels)

        self.enc1 = EncoderUnit(s_channels, 2 * s_channels)

        self.enc2 = EncoderUnit(2 * s_channels, 4 * s_channels)

        self.enc3 = EncoderUnit(4 * s_channels, 8 * s_channels)

        self.enc4 = EncoderUnit(8 * s_channels, 8 * s_channels)



        self.dec1 = DecoderUnit(16 * s_channels, 4 * s_channels)

        self.dec2 = DecoderUnit(8 * s_channels, 2 * s_channels)

        self.dec3 = DecoderUnit(4 * s_channels, s_channels)

        self.dec4 = DecoderUnit(2 * s_channels, s_channels)

        self.out = OutConv(s_channels, n_classes)



    def forward(self, x):

        x1 = self.conv(x)

        x2 = self.enc1(x1)

        x3 = self.enc2(x2)

        x4 = self.enc3(x3)

        x5 = self.enc4(x4)



        mask = self.dec1(x5, x4)

        mask = self.dec2(mask, x3)

        mask = self.dec3(mask, x2)

        mask = self.dec4(mask, x1)

        mask = self.out(mask)

        return mask, x5
import os

import cv2

import torch

import warnings

import ipywidgets

import numpy as np 

import pandas as pd

from torch import nn

import nibabel as nib

from glob import glob

from unet3d import UNet3d

from IPython import display

import matplotlib.pyplot as plt

from skimage.util import montage

from tqdm.autonotebook import tqdm

from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from keras.preprocessing.image import ImageDataGenerator

from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import StratifiedShuffleSplit
class Visulizer:

    def montage_nd(self, image):

        if len(image.shape)>3:

            return montage(np.stack([self.montage_nd(img) for img in image],0))

        elif len(image.shape)==3:

            return montage(image)

        else:

            warn('Input less than 3d image, returning original', RuntimeWarning)

            return image



    def visualize(self, image, mask):

        fig, axs = plt.subplots(1, 2, figsize = (20, 15 * 2))

        axs[0].imshow(self.montage_nd(image[..., 0]), cmap = 'bone')

        axs[1].imshow(self.montage_nd(mask[..., 0]), cmap = 'bone')

        plt.show()
viz = Visulizer()
class TrainDataset(Dataset):

    def __init__(self, BASE_PATH, num_slices = 64):

        self.num_slices = num_slices

        self.images = self.read_data(glob(os.path.join(BASE_PATH, "3d_images", "IMG_*")))

        self.masks = self.read_data(glob(os.path.join(BASE_PATH, "3d_images","MASK_*")), False)

        assert len(self.images) == len(self.masks)



    def read_data(self, paths, rescale = True, DS_FACT = 8):

        data = np.concatenate([nib.load(path).get_fdata()[:, ::DS_FACT, ::DS_FACT] for path in sorted(paths)], 0)

        if rescale: data = (data - data.min())/(data.max()-data.min()) * 255

        return np.expand_dims(data, -1).astype('float32') / 255



    def __len__(self):

        return len(self.images)-self.num_slices



    def __getitem__(self, idx):

        image = self.images[idx: idx + self.num_slices]

        mask = self.masks[idx: idx + self.num_slices]

        return image, mask
train_dataset = TrainDataset(os.path.join('/','kaggle','input', 'finding-lungs-in-ct-data'))
idx = np.random.choice(len(train_dataset))

image, mask = train_dataset[idx]

viz.visualize(image, mask)
class AugDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.aug_data_gen = ImageDataGenerator(

            rotation_range=15,

            width_shift_range=0.15,

            height_shift_range=0.15,

            shear_range=0.1,

            zoom_range=0.25,

            fill_mode='nearest',

            horizontal_flip=True,

            vertical_flip=False

        )



    def aug_data(self, x, y):

        xy = torch.cat([x, y], dim = 1).squeeze(dim = -1).permute(0, 2, 3, 1)

        img_gen = self.aug_data_gen.flow(xy, shuffle=True, seed=42, batch_size = len(x))

        # unblock

        xy_scat = torch.tensor(next(img_gen)).permute(0, 3, 1, 2).unsqueeze(dim = 1)

        return xy_scat[:, :, :xy_scat.shape[2]//2], xy_scat[:, :, xy_scat.shape[2]//2:]



    def __iter__(self):

        for data in super().__iter__():

            data = self.aug_data(*data)

            yield data
training = 0

num_epoch = 10

batch_size = 16

num_folds = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

warnings.filterwarnings("ignore")
checkpoint_path = os.path.join('/','kaggle','input', 'lung-segmentation-pytorch-unet3d', 'checkpoint.pth')

if os.path.exists(checkpoint_path):

    checkpoint = torch.load(checkpoint_path, map_location = device)

    torch.save(checkpoint, "checkpoint.pth")

else:

    checkpoint = {}
train_loaders = {}

valid_loaders = {}

train_folds = checkpoint.get('train_folds',{})

valid_folds = checkpoint.get('valid_folds',{})

sss = StratifiedShuffleSplit(n_splits = num_folds, test_size = 0.2, random_state = 42)

splitter = sss.split(train_dataset.images[:len(train_dataset)], train_dataset.masks[:len(train_dataset)].sum(1).sum(1).sum(1).astype('int')%64)

for fold, (train_indices, valid_indices) in enumerate(splitter):

    train_folds[fold] = train_folds.get(fold, train_indices)

    valid_folds[fold] = valid_folds.get(fold, valid_indices)

    # Creating PT data samplers and loaders

    train_sampler = SubsetRandomSampler(train_folds[fold])

    valid_sampler = SubsetRandomSampler(valid_folds[fold])

    train_loaders[fold] = AugDataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler)

    valid_loaders[fold] = AugDataLoader(train_dataset, batch_size = batch_size, sampler = valid_sampler)
fold = np.random.choice(list(train_loaders))

images, masks = next(iter(train_loaders[fold]))

viz.visualize(images.permute(0, 2, 3, 4, 1), masks.permute(0, 2, 3, 4, 1))
class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=0, logits=True, reduce=True):

        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.reduce = reduce

        self.loss_fn = (nn.BCEWithLogitsLoss if logits else nn.BCELoss)(reduction = 'none')



    def forward(self, pred, target):

        BCE_loss = self.loss_fn(pred, target)

        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return F_loss.mean() if self.reduce else F_loss
class DiceScore(nn.Module):

    def __init__(self, smooth = 1e-6):

        super().__init__()

        self.smooth = smooth



    def forward(self, pred, target):

        pred = torch.sigmoid(pred)

        batch_size = target.size(0)

        pred = pred.view(batch_size,-1)

        target = target.view(batch_size,-1)

        intersection = (pred * target)

        score = (2. * intersection.sum(1) + self.smooth) / (pred.sum(1) + target.sum(1) + self.smooth)

        return score.mean()
models = {}

optimizers = {}

schedulers = {}



metric = DiceScore()

criterion = FocalLoss()



for fold in tqdm(range(num_folds)):

    models[fold] = UNet3d(in_channels = 1, n_classes = 1, s_channels = 32).to(device)

    models[fold].load_state_dict(checkpoint.get("models", {}).get(fold, models[fold].state_dict()))

    # Prepare optimizer and schedule (linear warmup and decay)    

    params = [p for n, p in models[fold].named_parameters() if p.requires_grad]

    optimizers[fold] = torch.optim.Adam(params, lr=1e-3)

    optimizers[fold].load_state_dict(checkpoint.get("optimizers", {}).get(fold, optimizers[fold].state_dict()))

    schedulers[fold] = torch.optim.lr_scheduler.StepLR(optimizers[fold], step_size=90, gamma=0.1, last_epoch=-1)

    schedulers[fold].load_state_dict(checkpoint.get("schedulers", {}).get(fold, schedulers[fold].state_dict()))
def valid(fold):

    total_loss = 0

    total_score = 0

    models[fold].eval()

    loader = tqdm(valid_loaders[fold], desc = f"Validating fold {fold+1}")

    for idx, (images, targets) in enumerate(loader, start=1):

        images = images.to(device)

        targets = targets.to(device)

        # Execute

        with torch.no_grad():

            outputs, _ = models[fold](images)

        loss = criterion(outputs, targets).item()

        score = metric(outputs, targets).item()

        loss += 1 - score

        total_loss += loss

        total_score += score

        # print statistics

        loader.set_postfix_str(f"Score: {score:.4f} | Loss: {loss:.4f}")

        loader.update()

        # Clear variable

        del images; targets; del outputs; del loss; del score

    loader.write(f"Validated fold {fold+1} | Score: {total_score/idx:.4f} | Loss: {total_loss/idx:.4f}")

    return total_score/idx, total_loss/idx
def train(fold):

    total_loss = 0

    total_score = 0

    models[fold].train()

    loader = tqdm(train_loaders[fold], desc = f"Training fold {fold+1}")

    for idx, (images, targets) in enumerate(loader, start=1):

        images = images.to(device)

        targets = targets.to(device)

        # Execute

        outputs, _ = models[fold](images)

        loss = criterion(outputs, targets)

        score = metric(outputs, targets)

        loss += 1 - score

        total_loss += loss.item()

        total_score += score.item()

        # Optimize + Backward

        optimizers[fold].zero_grad()

        loss.backward()

        optimizers[fold].step()

        # print statistics

        loader.set_postfix_str(f"Score: {score:.4f} | Loss: {loss:.4f}")

        loader.update()

        # Clear variable

        del images; targets; del outputs; del loss; del score

    loader.write(f"Trained fold {fold+1} | Score: {total_score/idx:.4f} | Loss: {total_loss/idx:.4f}")

    valid_score, valid_loss = valid(fold)

    schedulers[fold].step()

    return total_score/idx, total_loss/idx, valid_score, valid_loss
train_data = checkpoint.get('train_data', {})

valid_data = checkpoint.get('valid_data', {})

epoch_data = checkpoint.get('epoch_data', [])

del checkpoint

fig, axs = plt.subplots(num_folds, 2, figsize=(10*2, 5*num_folds))

for fold in range(num_folds):

    if fold in valid_data and fold in train_data:

        # Visualize

        ax = axs[fold] if num_folds > 1 else axs

        ax[0].clear(); ax[1].clear()

        ax[0].plot(epoch_data, train_data[fold][:, 0], label = f"Train fold {fold+1} score {train_data[fold][-1, 0]:.4f}")

        ax[0].plot(epoch_data, valid_data[fold][:, 0], label = f"Valid fold {fold+1} score {valid_data[fold][-1, 0]:.4f}")

        ax[1].plot(epoch_data, train_data[fold][:, 1], label = f"Train fold {fold+1} loss {train_data[fold][-1, 1]:.4f}")

        ax[1].plot(epoch_data, valid_data[fold][:, 1], label = f"Valid fold {fold+1} loss {valid_data[fold][-1, 1]:.4f}")

        ax[0].legend(); ax[1].legend()

plt.show()
if training:

    loader = tqdm(range(len(epoch_data), len(epoch_data) + num_epoch), desc = "Epoch")

    board = ipywidgets.Output()

    display.display(board)

    graph = display.display(None, display_id = True)

    for i in loader:

        with board:

            epoch_data.append(i+1)

            # Make grid

            fig, axs = plt.subplots(num_folds, 2, figsize=(10*2, 5*num_folds))

            # Close figure

            plt.close(fig)

            for fold in range(num_folds):

                train_score, train_loss, valid_score, valid_loss = train(fold)

                train_data[fold] = np.append(train_data.get(fold, np.empty((0, 2))), [[train_score, train_loss]], axis = 0)

                valid_data[fold] = np.append(valid_data.get(fold, np.empty((0, 2))), [[valid_score, valid_loss]], axis = 0)

                # Visualize

                ax = axs[fold] if num_folds > 1 else axs

                ax[0].clear(); ax[1].clear()

                ax[0].plot(epoch_data, train_data[fold][:, 0], label = f"Train fold {fold+1} score {train_data[fold][-1, 0]:.4f}")

                ax[0].plot(epoch_data, valid_data[fold][:, 0], label = f"Valid fold {fold+1} score {valid_data[fold][-1, 0]:.4f}")

                ax[1].plot(epoch_data, train_data[fold][:, 1], label = f"Train fold {fold+1} loss {train_data[fold][-1, 1]:.4f}")

                ax[1].plot(epoch_data, valid_data[fold][:, 1], label = f"Valid fold {fold+1} loss {valid_data[fold][-1, 1]:.4f}")

                ax[0].legend(); ax[1].legend()

                graph.update(fig)

            # Clear all progress bar with in board widget

            display.clear_output()

            graph = display.display(fig, display_id = True)

        # Save model

        params = {

            'models': dict([(fold, models[fold].state_dict()) for fold in models]),

            'optimizers': dict([(fold, optimizers[fold].state_dict()) for fold in optimizers]),

            'schedulers': dict([(fold, schedulers[fold].state_dict()) for fold in schedulers]),

            'train_folds': train_folds,

            'valid_folds': valid_folds,

            'train_data': train_data,

            'valid_data': valid_data,

            'epoch_data': epoch_data

        }

        torch.save(params, "checkpoint.pth")

    loader.write("Done!")
idx = np.random.randint(len(train_dataset))

image, mask = train_dataset[idx]

img = torch.tensor(image, dtype = torch.float32).permute(3, 0, 1, 2).to(device)

pred = models[0](img[None].float())[0].detach().cpu()

pred = pred.squeeze(0).permute(1, 2, 3, 0)
viz.visualize(torch.sigmoid(pred), image)
viz.visualize(torch.sigmoid(pred), mask)
metric(torch.tensor(pred[None]), torch.tensor(mask[None])).item()