import os

import glob

import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm



Labels = pd.read_csv('/kaggle/input/faceimages/Face-Image-Eye-Annotations.csv')

Labels = Labels.sample(frac=1).reset_index(drop=True)

Labels.reset_index(drop=True, inplace=True)
for i in tqdm(range(Labels.shape[0])):

    image = cv2.imread('/kaggle/input/faceimages/Face-Images/Face-Images/'+Labels.iloc[i, 0], cv2.IMREAD_COLOR)

    Labels.loc[i, 'height'] = image.shape[0]

    Labels.loc[i, 'width'] = image.shape[1]
print(Labels.shape)



Labels = Labels[Labels.height >= 96]

print(Labels.shape)



Labels = Labels[Labels.Left_Eye_Center_X > 0]

Labels = Labels[Labels.Left_Eye_Center_Y > 0]

Labels = Labels[Labels.Right_Eye_Center_X > 0]

Labels = Labels[Labels.Right_Eye_Center_Y > 0]

print(Labels.shape)



Labels = Labels[Labels.Left_Eye_Center_X < Labels.width]

Labels = Labels[Labels.Left_Eye_Center_Y < Labels.height]

Labels = Labels[Labels.Right_Eye_Center_X < Labels.width]

Labels = Labels[Labels.Right_Eye_Center_Y < Labels.height]

print(Labels.shape)



Labels.reset_index(drop=True, inplace=True)

Labels
Labels.describe()
np.random.seed(42)

msk = np.random.rand(len(Labels)) < 0.9



Train_DF = Labels[msk].reset_index()

Validation_DF = Labels[~msk].reset_index()
Train_DF.describe()
Validation_DF.describe()
num_fig_rows = 5

num_fig_cols = 1



num_plots = num_fig_rows * num_fig_cols



rand_inds_vec = np.random.choice(Labels.shape[0],num_plots,replace=False)

rand_inds_mat = rand_inds_vec.reshape((num_fig_rows,num_fig_cols))



plt.close('all')

fig, ax = plt.subplots(nrows=num_fig_rows,ncols=num_fig_cols,figsize=(96,96))



for i in range(num_fig_rows):

    for j in range(num_fig_cols):

        curr_ind = rand_inds_mat[i][j]

        curr_image = cv2.imread('/kaggle/input/faceimages/Face-Images/Face-Images/'+Labels.iloc[curr_ind, 0], cv2.IMREAD_COLOR)

    

        x_feature_coords = np.array(Labels.iloc[curr_ind,[1,2]].tolist())

        y_feature_coords = np.array(Labels.iloc[curr_ind,[3,4]].tolist())

    

        ax[i].imshow(curr_image[:,:,[2,1,0]]);

        ax[i].scatter(x_feature_coords,y_feature_coords,c='r',s=20)

        ax[i].set_axis_off()

        ax[i].set_title('image index = '+(Labels.iloc[curr_ind, 0]),fontsize=10)
num_fig_rows = 5

num_fig_cols = 1



num_plots = num_fig_rows * num_fig_cols



rand_inds_vec = np.random.choice(1000,num_plots,replace=False)

rand_inds_mat = rand_inds_vec.reshape((num_fig_rows,num_fig_cols))



plt.close('all')

fig, ax = plt.subplots(nrows=num_fig_rows,ncols=num_fig_cols,figsize=(96,96))



for i in range(num_fig_rows):

    for j in range(num_fig_cols):

        curr_ind = rand_inds_mat[i][j]

        curr_image = cv2.imread('/kaggle/input/faceimages/Face-Images/Face-Images/'+Labels.iloc[curr_ind, 0], cv2.IMREAD_COLOR)

    

        x_feature_coords = np.array(Labels.iloc[curr_ind,[1,2]].tolist())

        y_feature_coords = np.array(Labels.iloc[curr_ind,[3,4]].tolist())

    

        ax[i].imshow(curr_image[:,:,[2,1,0]]);

        ax[i].scatter(x_feature_coords,y_feature_coords,c='r',s=20)

        ax[i].set_axis_off()

        ax[i].set_title('image index = '+(Labels.iloc[curr_ind, 0]),fontsize=10)
import torch

from torch.utils.data import Dataset, random_split, DataLoader

import torchvision.models as models

import torch.nn.functional as F

import torch.nn as nn

import torchvision.transforms as T



import albumentations as A

from albumentations.pytorch import ToTensorV2
class FaceImage(Dataset):

    

    def __init__(self, df, folder='/kaggle/input/faceimages/Face-Images/Face-Images/', transform=None):

        self.df = df

        self.transform = transform

        self.folder = folder

        

    def __len__(self):

        return len(self.df)    

    

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        image, lx, rx, ly, ry = row['Image'], row['Left_Eye_Center_X'], row['Right_Eye_Center_X'], row['Left_Eye_Center_Y'], row['Right_Eye_Center_Y']

        

        image = cv2.imread(self.folder + str(image))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        if self.transform:

            transformed = self.transform(image=image, keypoints=[(lx, ly), (rx, ry)])

            image = transformed["image"]

            keypoints = transformed["keypoints"]

#             lx, ly, rx, ry = keypoints[0][0], keypoints[0][1], keypoints[1][0], keypoints[1][1]

            

        return image, keypoints, lx, rx, ly, ry
train_transform = A.Compose(

    [

        A.Resize(128, 128),

#         A.HorizontalFlip(p=0.5),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ToTensorV2(),

    ],keypoint_params=A.KeypointParams(format='xy')

)



Train_DS = FaceImage(Train_DF, transform=train_transform)
valid_transform = A.Compose(

    [

        A.Resize(128, 128),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ToTensorV2(),

    ],keypoint_params=A.KeypointParams(format='xy')

)



Validation_DS = FaceImage(Validation_DF, transform=valid_transform)
Train_DS[0]
batch_size = 128



Train_DL = DataLoader(Train_DS, batch_size, shuffle=True, num_workers=4)

Validation_DL = DataLoader(Validation_DS, batch_size*2, shuffle=False, num_workers=4)
def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)



class DeviceDataLoader():

    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl: 

            yield to_device(b, self.device)



    def __len__(self):

        """Number of batches"""

        return len(self.dl)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
Train_DL = DeviceDataLoader(Train_DL, device)

Validation_DL = DeviceDataLoader(Validation_DL, device)
class ModelBase(nn.Module):

    

    def training_step(self, batch):

        images, targets, lx, rx, ly, ry = batch

        out = self(images)

        loss = 0.25*(F.mse_loss(targets[0][0], out[:,0]) + 

                     F.mse_loss(targets[0][1], out[:,1]) + 

                     F.mse_loss(targets[1][0], out[:,2]) + 

                     F.mse_loss(targets[1][1], out[:,3]))

        return loss

    

    def validation_step(self, batch):

        images, targets, lx, rx, ly, ry = batch

        out = self(images)

        loss = 0.25*(F.mse_loss(targets[0][0], out[:,0]) + 

                     F.mse_loss(targets[0][1], out[:,1]) + 

                     F.mse_loss(targets[1][0], out[:,2]) + 

                     F.mse_loss(targets[1][1], out[:,3]))

        return {'val_loss': loss.detach()}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()

        return {'val_loss': epoch_loss.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], last_lr: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}".format(

            epoch, result['lrs'][-1], result['train_loss'], result['val_loss']))
class EyeResNet(ModelBase):



    def __init__(self):

        super().__init__()

        # Use a pretrained model

        self.network = models.resnet50(pretrained=True)

        # Replace last layer

        num_ftrs = self.network.fc.in_features

        self.network.fc = nn.Linear(num_ftrs, 4)

    

    def forward(self, xb):

        return self.network(xb)

    

    def freeze(self):

        # To freeze the residual layers

        for param in self.network.parameters():

            param.require_grad = False

        for param in self.network.fc.parameters():

            param.require_grad = True

    

    def unfreeze(self):

        # Unfreeze all layers

        for param in self.network.parameters():

            param.require_grad = True
@torch.no_grad()

def evaluate(model, val_loader):

    model.eval()

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def get_lr(optimizer):

    for param_group in optimizer.param_groups:

        return param_group['lr']



def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 

                  weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):

    torch.cuda.empty_cache()

    history = []

    

    # Set up cutom optimizer with weight decay

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 

                                                steps_per_epoch=len(train_loader))

    

    for epoch in range(epochs):

        # Training Phase 

        model.train()

        train_losses = []

        lrs = []

        try:

            for batch in tqdm(train_loader):

                loss = model.training_step(batch)

                train_losses.append(loss)

                loss.backward()



                # Gradient clipping

                if grad_clip: 

                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)



                optimizer.step()

                optimizer.zero_grad()



                # Record & update learning rate

                lrs.append(get_lr(optimizer))

                sched.step()

                

        except:

            print('*'*10)

        

        # Validation phase

        result = evaluate(model, val_loader)

        result['train_loss'] = torch.stack(train_losses).mean().item()

        result['lrs'] = lrs

        model.epoch_end(epoch, result)

        history.append(result)

        

    return history
Model = EyeResNet().to(device)
[evaluate(Model, Validation_DL)]
Model.freeze()
epochs = 10

max_lr = 0.005

grad_clip = 0.1

weight_decay = 1e-4

opt_func = torch.optim.Adam
%%time

history = []

history += fit_one_cycle(epochs, max_lr, Model, Train_DL, Validation_DL, 

                         grad_clip=grad_clip, 

                         weight_decay=weight_decay, 

                         opt_func=opt_func)
torch.cuda.empty_cache()

images, targets, lx, rx, ly, ry = next(iter(Validation_DL))

Model.eval()

outputs = Model(images)
i = 20



Sample = images[i].permute(1,2,0).cpu().numpy()



lx = targets[0][0][i].cpu().numpy()

ly = targets[0][1][i].cpu().numpy()

rx = targets[1][0][i].cpu().numpy()

ry = targets[1][1][i].cpu().numpy()



plt.figure(figsize = (64, 64))

plt.imshow(Sample)



x_feature_coords = np.array([lx, rx])

y_feature_coords = np.array([ly, ry])

plt.scatter(x_feature_coords,y_feature_coords,c='r',s=500)



x_feature_coords = np.array([outputs[i].detach().cpu().numpy()[0], outputs[i].detach().cpu().numpy()[2]])

y_feature_coords = np.array([outputs[i].detach().cpu().numpy()[1], outputs[i].detach().cpu().numpy()[3]])

plt.scatter(x_feature_coords,y_feature_coords,c='b',s=500)
i = 40



Sample = images[i].permute(1,2,0).cpu().numpy()



lx = targets[0][0][i].cpu().numpy()

ly = targets[0][1][i].cpu().numpy()

rx = targets[1][0][i].cpu().numpy()

ry = targets[1][1][i].cpu().numpy()



plt.figure(figsize = (64, 64))

plt.imshow(Sample)



x_feature_coords = np.array([lx, rx])

y_feature_coords = np.array([ly, ry])

plt.scatter(x_feature_coords,y_feature_coords,c='r',s=500)



x_feature_coords = np.array([outputs[i].detach().cpu().numpy()[0], outputs[i].detach().cpu().numpy()[2]])

y_feature_coords = np.array([outputs[i].detach().cpu().numpy()[1], outputs[i].detach().cpu().numpy()[3]])

plt.scatter(x_feature_coords,y_feature_coords,c='b',s=500)
i = 60



Sample = images[i].permute(1,2,0).cpu().numpy()



lx = targets[0][0][i].cpu().numpy()

ly = targets[0][1][i].cpu().numpy()

rx = targets[1][0][i].cpu().numpy()

ry = targets[1][1][i].cpu().numpy()



plt.figure(figsize = (64, 64))

plt.imshow(Sample)



x_feature_coords = np.array([lx, rx])

y_feature_coords = np.array([ly, ry])

plt.scatter(x_feature_coords,y_feature_coords,c='r',s=500)



x_feature_coords = np.array([outputs[i].detach().cpu().numpy()[0], outputs[i].detach().cpu().numpy()[2]])

y_feature_coords = np.array([outputs[i].detach().cpu().numpy()[1], outputs[i].detach().cpu().numpy()[3]])

plt.scatter(x_feature_coords,y_feature_coords,c='b',s=500)
i = 80



Sample = images[i].permute(1,2,0).cpu().numpy()



lx = targets[0][0][i].cpu().numpy()

ly = targets[0][1][i].cpu().numpy()

rx = targets[1][0][i].cpu().numpy()

ry = targets[1][1][i].cpu().numpy()



plt.figure(figsize = (64, 64))

plt.imshow(Sample)



x_feature_coords = np.array([lx, rx])

y_feature_coords = np.array([ly, ry])

plt.scatter(x_feature_coords,y_feature_coords,c='r',s=500)



x_feature_coords = np.array([outputs[i].detach().cpu().numpy()[0], outputs[i].detach().cpu().numpy()[2]])

y_feature_coords = np.array([outputs[i].detach().cpu().numpy()[1], outputs[i].detach().cpu().numpy()[3]])

plt.scatter(x_feature_coords,y_feature_coords,c='b',s=500)
weights_fname = 'iris-resnet-1.pth'

torch.save(Model.state_dict(), weights_fname)