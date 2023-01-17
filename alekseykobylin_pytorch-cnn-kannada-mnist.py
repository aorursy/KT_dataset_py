import os

import math

import random

import torch

import torchvision

import torchvision.transforms as transforms

import torchvision.transforms.functional as F

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt



from tqdm.autonotebook import tqdm





METRICS = {

    'accuracy': {

        'f': accuracy_score,

        'args': {}

    },

    # 'balanced_accuracy': {

    #     'f': balanced_accuracy_score,

    #     'args': {}

    # },

    # 'f1': {

    #     'f': f1_score,

    #     'args': {'average': 'weighted'}

    # },

    # 'precision': {

    #     'f': precision_score,

    #     'args': {'average': 'weighted'}

    # },

    # 'recall': {

    #     'f': recall_score,

    #     'args': {'average': 'weighted'}

    # }

}





NORM_MEAN = [0.485, 0.456, 0.406]

NORM_STD = [0.229, 0.224, 0.225]





def make_image_label_grid(images, labels=None, class_names=None):

    channels = images.shape[1]

    if channels not in (3, 1):

        raise ValueError("Images must have 1 or 3 channels")

    mean = NORM_MEAN if channels == 3 else [sum(NORM_MEAN) / 3]

    std = NORM_STD if channels == 3 else [sum(NORM_STD) / 3]

    mean = torch.tensor(mean)

    std = torch.tensor(std)

    mean = (-mean / std).tolist()

    std = (1.0 / std).tolist()

    img_grid = torchvision.utils.make_grid(images)

    img_grid = F.normalize(img_grid, mean=mean, std=std)

    return img_grid





def make_image_label_figure(images, labels=None, class_names=None):

    channels = images.shape[1]

    if channels not in (3, 1):

        raise ValueError("Images must have 1 or 3 channels")

    mean = NORM_MEAN if channels == 3 else [sum(NORM_MEAN) / 3]

    std = NORM_STD if channels == 3 else [sum(NORM_STD) / 3]

    mean = torch.tensor(mean)

    std = torch.tensor(std)

    mean = (-mean / std).tolist()

    std = (1.0 / std).tolist()

    n = int(math.sqrt(len(images)))

    figure = plt.figure(figsize=(n, n))

    figure.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n*n):

        image, label = images[i], (0 if labels is None else labels[i])

        image = F.normalize(image, mean=mean, std=std)

        image = image.permute(1, 2, 0)

        image = torch.squeeze(image)

        image = (image * 255).int()

        plt.subplot(n, n, i + 1, title='NA' if class_names is None else class_names[label])

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(image, cmap='gray' if channels == 1 else None)

    return figure





class Transforms(transforms.Compose):



    def __init__(self, in_channels=1, out_channels=1, size=(32, 32)):

        if out_channels not in (3, 1) or in_channels not in (3, 1):

            raise ValueError("Images must have 1 or 3 channels")

        mean = NORM_MEAN if out_channels == 3 else [sum(NORM_MEAN) / 3]

        std = NORM_STD if out_channels == 3 else [sum(NORM_STD) / 3]

        transforms_list = []

        if in_channels != out_channels:

            transforms_list.append(transforms.Grayscale(out_channels))

        transforms_list.extend([

            transforms.Resize(size),

            transforms.ToTensor(),

            transforms.Normalize(mean=mean, std=std)])

        super(Transforms, self).__init__(transforms_list)





class TrainTransforms(transforms.Compose):



    def __init__(self, in_channels=1, out_channels=1, size=(32, 32), random_crop=False, random_affine=False,

                 horizontal_flip=False, color_jitter=False, random_erasing=False):

        if out_channels not in (3, 1) or in_channels not in (3, 1):

            raise ValueError("Images must have 1 or 3 channels")

        mean = NORM_MEAN if out_channels == 3 else [sum(NORM_MEAN) / 3]

        std = NORM_STD if out_channels == 3 else [sum(NORM_STD) / 3]

        transforms_list = []

        if in_channels != out_channels:

            transforms_list.append(transforms.Grayscale(out_channels))

        if random_affine:

            transforms_list.append(transforms.RandomAffine(degrees=10.0, translate=(0.25, 0.25),

                                                           shear=(-10, 10, -10, 10)))

        if random_crop:

            transforms_list.append(transforms.RandomResizedCrop(size, scale=(0.9, 1.1), ratio=(0.75, 1.33)))

        else:

            transforms_list.append(transforms.Resize(size))

        if horizontal_flip:

            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if color_jitter:

            transforms_list.append(transforms.ColorJitter(brightness=(0.75, 1.5), contrast=(0.75, 1.5),

                                                          saturation=(0.75, 1.5), hue=(-0.1, 0.1)))

        transforms_list.append(transforms.ToTensor())

        if random_erasing:

            transforms_list.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0))

        transforms_list.append(transforms.Normalize(mean=mean, std=std))

        super(TrainTransforms, self).__init__(transforms_list)





class TrainerProgressBar(tqdm):



    def __init__(self, desc=None, total=10, unit='it', position=None):

        super(TrainerProgressBar, self).__init__(

            desc=desc, total=total, leave=True, unit=unit, position=position, dynamic_ncols=True

        )



    def reset(self, total=None, desc=None, ordered_dict=None):

        # super(TrainerProgressBar, self).reset(total)

        self.last_print_n = self.n = 0

        self.last_print_t = self.start_t = self._time()

        if total is not None:

            self.total = total

        super(TrainerProgressBar, self).refresh()

        if desc is not None:

            super(TrainerProgressBar, self).set_description(desc)

        if ordered_dict is not None:

            super(TrainerProgressBar, self).set_postfix(ordered_dict)



    def update(self, desc=None, ordered_dict=None, n=1):

        if desc is not None:

            super(TrainerProgressBar, self).set_description(desc)

        if ordered_dict is not None:

            super(TrainerProgressBar, self).set_postfix(ordered_dict)

        super(TrainerProgressBar, self).update(n)





class PyTorchTrainer(object):



    def __init__(self, device=None, metrics=None, epoch_callback=None, batch_callback=None):

        self.device = torch.device(device) if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.metrics = metrics or METRICS

        self.epoch_callback = epoch_callback

        self.batch_callback = batch_callback



        self.train_pb = None

        self.epoch_train_pb = None

        self.epoch_val_pb = None



    def train(self, model, optimizer, loss_criterion, train_data_loader, val_data_loader, scheduler=None,

              epochs=10):

        # Print log

        print(f"=============================== Training NN ===============================")

        print(f"== Epochs:              {epochs:6d}")

        print(f"== Train batch size:    {train_data_loader.batch_size:6d}")

        print(f"== Train batches:       {len(train_data_loader):6d}")

        print(f"== Validate batch size: {val_data_loader.batch_size:6d}")

        print(f"== Validate batches:    {len(val_data_loader):6d}")

        print(f"===========================================================================")

        # Initialize progress bars

        self.train_pb = TrainerProgressBar(desc=f'== Epoch {1}', total=epochs, unit='epoch', position=0)

        self.epoch_train_pb = TrainerProgressBar(desc=f'== Train {1}', total=len(train_data_loader),

                                                 unit='batch', position=1)

        self.epoch_val_pb = TrainerProgressBar(desc=f'== Val {1}', total=len(val_data_loader), unit='batch',

                                               position=2)

        # Reset progress bars

        self.train_pb.reset(total=epochs)

        self.epoch_train_pb.reset(total=len(train_data_loader))

        self.epoch_val_pb.reset(total=len(val_data_loader))

        for epoch in range(epochs):

            # Train batches

            train_loss, predictions, targets, inputs = self.forward_batches(model, optimizer, loss_criterion,

                                                                            train_data_loader, epoch, train=True)

            # Val batches

            val_loss, predictions, targets, inputs = self.forward_batches(model, optimizer, loss_criterion,

                                                                          val_data_loader, epoch, train=False)

            # Make scheduler step

            if scheduler:

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):

                    scheduler.step(train_loss, epoch=epoch)

                else:

                    scheduler.step(epoch=epoch)

            # Update progress bar

            metrics_dict = {'lr': optimizer.param_groups[0]['lr'],

                            'loss': {'train': round(train_loss, 4), 'val': round(val_loss, 4)}}

            metrics_dict.update(self.metrics_dict(predictions, targets))

            if self.epoch_callback:

                self.epoch_callback(epoch, epochs, predictions, targets, inputs, metrics_dict)

            self.train_pb.update(desc=f'== Epoch {epoch+1}', ordered_dict=metrics_dict)

        # Close progress bars

        self.train_pb.close()

        self.epoch_train_pb.close()

        self.epoch_val_pb.close()

        # Print log

        print(f"===========================================================================")



    def forward_batches(self, model, optimizer, loss_criterion, data_loader, epoch, train=True):

        # Set model train or eval due to current phase

        if train:

            model.train()

        else:

            model.eval()

        # Preset variables

        avg_loss_value = 0

        all_predictions = None

        all_targets = None

        all_inputs = None

        batches = len(data_loader)

        # Reset progress bar

        if train:

            self.epoch_train_pb.reset(batches, f"== Train {epoch+1}")

        else:

            self.epoch_val_pb.reset(batches, f"== Val {epoch+1}")

        for batch_i, data in enumerate(data_loader, 1):

            # Forward batch

            loss_value, predictions, targets, inputs = self.forward_batch(model, optimizer, loss_criterion, data,

                                                                          train=train)

            # Update variables

            avg_loss_value += loss_value

            all_predictions = predictions if all_predictions is None else torch.cat((all_predictions, predictions))

            all_targets = targets if all_targets is None else torch.cat((all_targets, targets))

            all_inputs = inputs if all_inputs is None else torch.cat((all_inputs, inputs))

            # Update progress bar

            metrics_dict = {'lr': optimizer.param_groups[0]['lr'], 'loss': avg_loss_value/batch_i}

            metrics_dict.update(self.metrics_dict(all_predictions, all_targets))

            if self.batch_callback:

                self.batch_callback(train, epoch, batch_i, batches, predictions, targets, inputs, metrics_dict)

            if train:

                self.epoch_train_pb.update(ordered_dict=metrics_dict)

            else:

                self.epoch_val_pb.update(ordered_dict=metrics_dict)

        # Update variables

        avg_loss_value /= batches

        # Return

        return avg_loss_value, all_predictions, all_targets, all_inputs



    def forward_batch(self, model, optimizer, loss_criterion, batch_data, train=True):

        # Get Inputs and Targets and put them to device

        inputs, targets = batch_data

        inputs = inputs.to(self.device)

        targets = targets.to(self.device)

        with torch.set_grad_enabled(train):

            # Forward model to get outputs

            outputs = model.forward(inputs)

            # Calculate Loss Criterion

            loss = loss_criterion(outputs, targets)

        if train:

            # Zero optimizer gradients

            optimizer.zero_grad()

            # Calculate new gradients

            loss.backward()

            # Make optimizer step

            optimizer.step()

        # Variables

        loss_value = loss.item()

        predictions = outputs.argmax(dim=1).data.cpu()

        targets = targets.data.cpu()

        inputs = inputs.data.cpu()

        return loss_value, predictions, targets, inputs



    def metrics_dict(self, predictions, targets):

        d = {}

        for metric_name in self.metrics:

            metric_value = self.metrics[metric_name]['f'](predictions, targets, **self.metrics[metric_name]['args'])

            d[metric_name] = metric_value



        return d
import os

import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler

import torchvision.transforms as transforms

from PIL import Image
CLASS_NAMES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

data_root = '../input/Kannada-MNIST'

train_file_name = 'train.csv'

val_file_name = 'Dig-MNIST.csv'

test_file_name = 'test.csv'

NORM_MEAN = [0.485, 0.456, 0.406]

NORM_STD = [0.229, 0.224, 0.225]
# Transforms

class KannadaMNISTTransforms(transforms.Compose):



    def __init__(self, in_channels=1, out_channels=1, size=(28, 28)):

        if out_channels not in (3, 1) or in_channels not in (3, 1):

            raise ValueError("Images must have 1 or 3 channels")

        mean = NORM_MEAN if out_channels == 3 else [sum(NORM_MEAN) / 3]

        std = NORM_STD if out_channels == 3 else [sum(NORM_STD) / 3]

        transforms_list = []

        if in_channels != out_channels:

            transforms_list.append(transforms.Grayscale(out_channels))

        transforms_list.extend([

            transforms.RandomAffine(10.0, translate=(0.25, 0.25), shear=(-10, 10, -10, 10)),

            transforms.Resize(size),

            transforms.ToTensor(),

            transforms.RandomErasing(p=0.2, scale=(0.05, 0.1), ratio=(0.3, 3.3), value=0), 

            transforms.Normalize(mean=mean, std=std)

        ])

        super(KannadaMNISTTransforms, self).__init__(transforms_list)
# Dataset

class KannadaMNISTDataset(torch.utils.data.Dataset):

    

    def __init__(self, images, targets=None, transform=None):

        super(KannadaMNISTDataset, self).__init__()

        self.images = [Image.fromarray(image) for image in images]

        self.targets = np.zeros(len(images)) if targets is None else targets.astype(int)

        self.transform = transform

    

    def __len__(self):

        return len(self.images)

        

    def __getitem__(self, index):

        return self.images[index] if self.transform is None else self.transform(self.images[index]), torch.tensor(self.targets[index], dtype=torch.long)
# Read *.csv files to Pandas Dataframes

train_df = pd.read_csv(os.path.join(data_root, train_file_name))

val_df = pd.read_csv(os.path.join(data_root, val_file_name))

test_df = pd.read_csv(os.path.join(data_root, test_file_name))
# Construct from DataFrames train/val/test datasets

train_targets = train_df.label.values.astype(int)

train_images = (train_df.drop('label', axis=1).values.astype(float) / 255).reshape(-1, 28, 28)

val_targets = val_df.label.values.astype(int)

val_images = (val_df.drop('label', axis=1).values.astype(float) / 255).reshape(-1, 28, 28)

test_ids = test_df.id.values.astype(int)

test_images = (test_df.drop('id', axis=1).values.astype(float) / 255).reshape(-1, 28, 28)

print("Origin trainset images/targets shapes", train_images.shape, train_targets.shape)

print("Origin valset images/targets shapes", val_images.shape, val_targets.shape)

print("Origin testset images/ids shape", test_images.shape, test_ids.shape)

print()
# Train/val/test dataloaders and datasets

# Batch size

batch_size = 1000

size = (28, 28)

origin_channels = 1

in_channels = 1

# Datasets

train_dataset = KannadaMNISTDataset(train_images, train_targets, transform=KannadaMNISTTransforms(in_channels=origin_channels, out_channels=in_channels, size=size))

val_dataset = KannadaMNISTDataset(val_images, val_targets, transform=Transforms(in_channels=origin_channels, out_channels=in_channels, size=size))

test_dataset = KannadaMNISTDataset(test_images, transform=Transforms(in_channels=origin_channels, out_channels=in_channels, size=size))

# Dataloaders

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# Show train batch

images, targets = next(iter(train_dataloader))

fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)
# Show val batch

images, targets = next(iter(val_dataloader))

fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)
# Show test batch

images = next(iter(test_dataloader))

fig = make_image_label_figure(images[0][:9])
class Conv2dBNReLU(nn.Sequential):

    '''Convolution2d + BatchNormalization2d + ReLU Activation'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):

        super(Conv2dBNReLU, self).__init__(

            nn.Conv2d(in_channels, out_channels,

                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )





class DOLinearBNReLU(nn.Sequential):

    '''Droupout + Linear'''

    def __init__(self, in_features, out_features, bias=True):

        super(DOLinearBNReLU, self).__init__(

            nn.Dropout(0.2),

            nn.Linear(in_features, out_features, bias=bias),

            nn.BatchNorm1d(out_features),

            nn.ReLU(inplace=True)

        )





class DOLinear(nn.Sequential):

    '''Droupout + Linear'''

    def __init__(self, in_features, out_features, bias=True):

        super(DOLinear, self).__init__(

            nn.Dropout(0.2),

            nn.Linear(in_features, out_features, bias=bias)

        )





class KannadaMNISTNet(nn.Module):



    def __init__(self, in_channels=1, classes=10):

        super(KannadaMNISTNet, self).__init__()



        self.feature_extractor = nn.Sequential(

            Conv2dBNReLU(in_channels, 64, kernel_size=3, padding=1),

            

            Conv2dBNReLU(64, 64, kernel_size=3, padding=1),

            Conv2dBNReLU(64, 64, kernel_size=3, padding=1),



            nn.MaxPool2d(2),



            Conv2dBNReLU(64, 128, kernel_size=3, padding=1),

            Conv2dBNReLU(128, 128, kernel_size=3, padding=1),

            Conv2dBNReLU(128, 128, kernel_size=3, padding=1),



            nn.MaxPool2d(2),



            Conv2dBNReLU(128, 256, kernel_size=3, padding=1),

            Conv2dBNReLU(256, 256, kernel_size=3, padding=1),



            Conv2dBNReLU(256, 512, kernel_size=3, padding=1),

            Conv2dBNReLU(512, 512, kernel_size=3, padding=1),

        )

        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(

            DOLinearBNReLU(7 * 7 * 512, 512),

            DOLinear(512, classes)

        )

        self.sm = nn.Softmax()



    def forward(self, x):

        x = self.feature_extractor(x)

        x = self.pool(x)

        x = x.view(-1, 7 * 7 * 512)

        x = self.classifier(x)

        return x



    def predict(self, x):

        x = self.forward(x)

        x = self.sm(x)

        return x
# Train Net

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



net = KannadaMNISTNet(in_channels=in_channels, classes=10)

net = net.to(torch.device(device))



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.002, weight_decay=0.00005)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2, verbose=False, mode='min', threshold=0.0001, cooldown=0, min_lr=0.000001)



trainer = PyTorchTrainer(device=device)

trainer.train(net, optimizer, criterion, train_dataloader, val_dataloader, scheduler=scheduler, epochs=100)
# Trigger net to eval mode

net.eval()

for param in net.parameters():

    param.requires_grad = False
# Check some validation dataset predictions

images, targets = next(iter(val_dataloader))

_images = images.to(torch.device(device))

_output = net.forward(_images)

predictions = _output.argmax(dim=1).data.cpu()



fig = make_image_label_figure(images[:9], predictions[:9], CLASS_NAMES)
# Check some test dataset predictions

images = next(iter(test_dataloader))[0]

_images = images.to(torch.device(device))

_output = net.forward(_images)

predictions = _output.argmax(dim=1).data.cpu()



fig = make_image_label_figure(images[:9], predictions[:9], CLASS_NAMES)
# Get all test predictions

test_predictions = None

for batch in test_dataloader:

    inputs = batch[0]

    inputs = inputs.to(torch.device(device))

    output = net.forward(inputs)

    predictions = output.argmax(dim=1).data

    test_predictions = predictions if test_predictions is None else torch.cat((test_predictions, predictions))

test_predictions = test_predictions.cpu().numpy()
# Write submission to pd.DataFrame

submission_df = pd.DataFrame(np.c_[test_ids[:,None], test_predictions], columns=['id', 'label'])

submission_df.head()
# Write submission DataFrame to csv

submission_df.to_csv('submission.csv', index=False)