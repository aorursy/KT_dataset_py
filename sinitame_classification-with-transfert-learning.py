from glob import glob

import pandas as pd

import torch, pydicom, torchvision, time, os, cv2, copy

import torch.utils.data

import torch.utils.data.dataloader

from collections import defaultdict, deque

from torchvision import transforms, utils

from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

import matplotlib.pyplot as plt
input_size = 512



class SIIMDataset(torch.utils.data.Dataset):

    """SIIM Pneumothorax dataset."""



    def __init__(self, csv_file, root_dir, transform=None):

        self.masks = pd.read_csv(csv_file)

        self.masks = self.masks.drop_duplicates('ImageId', keep='last').reset_index(drop=True)

        self.images = glob(os.path.join(root_dir,'*/*/*.dcm'))

        self.transform = transform

        self.width = input_size

        self.height = input_size



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        img_path = self.images[idx]

        img_file_name = os.path.basename(img_path)

        img_name = os.path.splitext(img_file_name)[0]

        image_infos = pydicom.dcmread(img_path)

        

        image_full = np.expand_dims(image_infos.pixel_array, axis=2)

        image_cv2 = cv2.resize(image_full, (input_size,input_size), interpolation=cv2.INTER_CUBIC)

        image = np.expand_dims(image_cv2, axis=2)

        

        mask_serie = self.masks.loc[self.masks["ImageId"] == img_name]

        mask_rle = mask_serie[" EncodedPixels"].tolist()

        mask = np.zeros((input_size, input_size))

        img_class = np.zeros((1))

        

        for mask_i in mask_rle:

            if mask_i != " -1":

                mask += rle2mask(mask_i, self.width, self.height)

                img_class = np.array([1])

            else:

                img_class = np.array([0])



        sample = {"image": image , "mask": mask, "class":img_class }





        if self.transform:

            sample = self.transform(sample)



        return sample



    

class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""



    def __call__(self, sample):

        image, mask, img_class = sample['image'], sample['mask'], sample['class']



        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W

        image_np = image.transpose((2, 0, 1))

        image = np.repeat(image_np , 3, 0)

        sample['image'] = torch.from_numpy(image).float()

        sample['mask'] = torch.from_numpy(mask).float()

        sample['class'] = torch.from_numpy(img_class).long()

        return sample 
img = '../input/pneumothorax/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.5296.1517875187.462352/1.2.276.0.7230010.3.1.3.8323329.5296.1517875187.462351/1.2.276.0.7230010.3.1.4.8323329.5296.1517875187.462353.dcm'

img_file_name = os.path.basename(img)

img_name = os.path.splitext(img_file_name)[0]

image_infos = pydicom.dcmread(img)

print("ok")

image_full = np.expand_dims(image_infos.pixel_array, axis=2)

print(image_full.shape)

image = cv2.resize(image_full, (input_size,input_size), interpolation=cv2.INTER_CUBIC)

img1 = np.expand_dims(image, axis=2)

img2 = img1.transpose((2, 0, 1)) 

img3 = np.repeat(img2, 3, 0)

print( img2.shape)

print(img3.shape)
def rle2mask(rle, width, height):

    mask= np.zeros(width* height)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        mask[current_position:current_position+lengths[index]] = 1

        current_position += lengths[index]



    return mask.reshape(width, height)


from albumentations import (

    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,

    IAASharpen, IAAEmboss, Flip, OneOf, Compose

)



def strong_aug(p=0.5):

    return Compose([

        RandomRotate90(),

        Flip(),

        Transpose(),

        OneOf([

            IAAAdditiveGaussianNoise(),

            GaussNoise(),

        ], p=0.2),

        OneOf([

            MotionBlur(p=0.2),

            MedianBlur(blur_limit=3, p=0.1),

            Blur(blur_limit=3, p=0.1),

        ], p=0.2),

        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),

        OneOf([

            OpticalDistortion(p=0.3),

            GridDistortion(p=0.1),

            IAAPiecewiseAffine(p=0.3),

        ], p=0.2),

        OneOf([

            CLAHE(clip_limit=2),

            IAASharpen(),

            IAAEmboss(),

            RandomBrightnessContrast(),

        ], p=0.3),

        HueSaturationValue(p=0.3),

    ], p=p)













dataset = SIIMDataset('../input/pneumothorax/train-rle.csv','../input/pneumothorax/dicom-images-train', transform=data_transform)
num_workers = 4

batch_size = 16

validation_split = 0.2



dataset_size = len(dataset)

indices = list(range(dataset_size))

split = int(np.floor(validation_split * dataset_size))

train_indices, val_indices = indices[split:], indices[:split]



# Creating PT data samplers and loaders:

train_sampler = SubsetRandomSampler(train_indices)

valid_sampler = SubsetRandomSampler(val_indices)



train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)

validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,sampler=valid_sampler)



dataloaders = {'train': train_loader,'val': validation_loader}
import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



#Importing Resnet50 with pretrained weights

model = torchvision.models.resnet50(pretrained=True)



for param in model.parameters():

    param.requires_grad = False



fc_in = model.fc.in_features

fc_layers = nn.Sequential(

                nn.Linear(fc_in, 2),

                nn.Dropout(0.3),

                nn.Softmax(),

            )







model.fc = fc_layers

model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):

    since = time.time()



    val_acc_history = []



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

            for i,data in enumerate(dataloaders[phase]):

                inputs = data['image'].to(device)

                labels = data['class'].to(device)

                

                labels = labels.squeeze(1)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss

                    

                    outputs = model(inputs)

                    if (i<5):

                        print(labels)

                        print(torch.max(outputs,1))

                    loss = criterion(outputs, labels)



                    _, preds = torch.max(outputs, 1)

                    



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':

                val_acc_history.append(epoch_acc)



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model, val_acc_history
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

torch.cuda.empty_cache()

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),  lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

model_final, val_acc_history = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)