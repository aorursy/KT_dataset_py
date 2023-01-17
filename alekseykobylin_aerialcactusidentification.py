!mkdir data

!cp ../input/aerial-cactus-identification/* data
!unzip -o data/test.zip -d data >/dev/null

!unzip -o data/train.zip -d data >/dev/null

!rm -f data/*.zip
import os

import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler

import torchvision.transforms as transforms

import torchvision

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from PIL import Image
CLASS_NAMES = ('0', '1')

data_root = './data'

train_data_path = os.path.join(data_root, 'train')

test_data_path = os.path.join(data_root, 'test')

train_set_path = os.path.join(train_data_path, 'train')

dev_set_path = os.path.join(train_data_path, 'dev')

test_set_path = os.path.join(train_data_path, 'test')

NORM_MEAN = [0.485, 0.456, 0.406]

NORM_STD = [0.229, 0.224, 0.225]
# Read csv file of train set and create right data

train_targets = pd.read_csv(os.path.join(data_root, 'train.csv'))
X = train_targets['id'].values

y = train_targets['has_cactus'].values.astype(int)
from sklearn.model_selection import train_test_split

X_train, _X_test, y_train, _y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y)

X_dev, X_test, y_dev, y_test = train_test_split(_X_test, _y_test, test_size=0.5, shuffle=True, stratify=_y_test)

no_cactus_weight = (y_train==1).sum() / y_train.shape[0]

has_cactus_weight = (y_train==0).sum() / y_train.shape[0]
for set_path in (train_set_path, dev_set_path, test_set_path):

  os.system(f"mkdir {set_path}")

  for class_name in CLASS_NAMES:

    os.system(f"mkdir {os.path.join(set_path, class_name)}")
for path, file_names, labels in ((train_set_path, X_train, y_train), (dev_set_path, X_dev, y_dev), (test_set_path, X_test, y_test)):

  for file_name, label in zip(file_names, labels):

    os.system(f"mv -f {os.path.join(train_data_path, file_name)} {os.path.join(path, str(label), file_name)}")
import os

import math

import random

import torch

import torchvision

import torchvision.transforms as transforms

import torchvision.transforms.functional as F

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter



from tqdm.autonotebook import tqdm





METRICS = {

    'accuracy': {

        'f': balanced_accuracy_score,

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

            train_loss, train_metrics_dict = self.forward_batches(model, optimizer, loss_criterion,

                                                                  train_data_loader, epoch, train=True)

            # Val batches

            val_loss, val_metrics_dict = self.forward_batches(model, optimizer, loss_criterion,

                                                              val_data_loader, epoch, train=False)

            # Make scheduler step

            if scheduler:

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):

                    scheduler.step(train_loss, epoch=epoch)

                else:

                    scheduler.step(epoch=epoch)

            # Update progress bar

            lr = optimizer.param_groups[0]['lr']

            if self.epoch_callback:

                self.epoch_callback(model, epoch, lr, train_loss, val_loss, train_metrics_dict, val_metrics_dict)

            metrics_dict = {'lr': lr}

            for metric_name in train_metrics_dict:

                metrics_dict[f"train_{metric_name}"] = train_metrics_dict[metric_name]

                metrics_dict[f"val_{metric_name}"] = val_metrics_dict[metric_name]

            metrics_dict.update({'train_loss': train_loss, 'val_loss': val_loss})

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

        avg_metrics_dict = None

        batches = len(data_loader)

        # Reset progress bar

        if train:

            self.epoch_train_pb.reset(batches, f"== Train {epoch+1}")

        else:

            self.epoch_val_pb.reset(batches, f"== Val {epoch+1}")

        for batch_i, data in enumerate(data_loader, 1):

            # Forward batch

            loss_value, predictions, targets = self.forward_batch(model, optimizer, loss_criterion, data,

                                                                  train=train)

            metrics_dict = self.metrics_dict(predictions, targets)

            # Update variables

            avg_loss_value += loss_value

            if avg_metrics_dict is None:

                avg_metrics_dict = metrics_dict.copy()

            else:

                for metric_name in avg_metrics_dict:

                    avg_metrics_dict[metric_name] += metrics_dict[metric_name]

            # Update progress bar

            if self.batch_callback:

                self.batch_callback(train, epoch, batch_i, batches, loss_value, metrics_dict)

            metrics_dict.update({'loss': avg_loss_value/batch_i})

            if train:

                self.epoch_train_pb.update(ordered_dict=metrics_dict)

            else:

                self.epoch_val_pb.update(ordered_dict=metrics_dict)

        # Update variables

        avg_loss_value /= batches

        for metric_name in avg_metrics_dict:

            avg_metrics_dict[metric_name] /= batches

        # Return

        return avg_loss_value, avg_metrics_dict



    def forward_batch(self, model, optimizer, loss_criterion, batch_data, train=True):

        # Get Inputs and Targets and put them to device

        inputs, targets = batch_data

        _inputs = inputs.to(self.device)

        _targets = targets.to(self.device)

        with torch.set_grad_enabled(train):

            # Forward model to get outputs

            _outputs = model.forward(_inputs)

            # Calculate Loss Criterion

            loss = loss_criterion(_outputs, _targets)

        if train:

            # Zero optimizer gradients

            optimizer.zero_grad()

            # Calculate new gradients

            loss.backward()

            # Make optimizer step

            optimizer.step()

        # Variables

        loss_value = loss.item()

        predictions = _outputs.argmax(dim=1).data.cpu()

        targets = targets.data

        return loss_value, predictions, targets



    def metrics_dict(self, predictions, targets):

        d = {}

        for metric_name in self.metrics:

            metric_value = self.metrics[metric_name]['f'](predictions, targets, **self.metrics[metric_name]['args'])

            d[metric_name] = metric_value



        return d
# Transforms

class TrainTransforms(transforms.Compose):



    def __init__(self):

        super(TrainTransforms, self).__init__([

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomVerticalFlip(p=0.5),

#             transforms.ColorJitter(brightness=(0.75, 1.5), contrast=(0.75, 1.5), 

#                                    saturation=(0.75, 1.5), hue=(-0.1, 0.1)),

            transforms.ToTensor(),

            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

        ])





class TestTransforms(transforms.Compose):



    def __init__(self):

        super(TestTransforms, self).__init__([

            transforms.ToTensor(),

            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

        ])
# Train/dev/test dataloaders and datasets

# Batch size

batch_size = 1024

# Datasets

train_dataset = torchvision.datasets.ImageFolder(train_set_path, transform=TrainTransforms())

dev_dataset = torchvision.datasets.ImageFolder(dev_set_path, transform=TestTransforms())

test_dataset = torchvision.datasets.ImageFolder(test_set_path, transform=TestTransforms())

# Dataloaders

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# Show train batch

images, targets = next(iter(train_dataloader))

fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)
# Show dev batch

images, targets = next(iter(dev_dataloader))

fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)
# Show test batch

images, targets = next(iter(test_dataloader))

fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)
import torchvision.models as models





class ResNet(nn.Module):



  def __init__(self, resnet18, classes):

    super(ResNet, self).__init__()

    self.feature_extractor = nn.Sequential(

      resnet18.conv1,

      resnet18.bn1,

      resnet18.relu,

      resnet18.maxpool,

      resnet18.layer1,

      resnet18.layer2,

      resnet18.layer3

    )

    self.classifier = nn.Sequential(

      nn.Dropout(0.2),

      nn.Linear(2*2*256, classes)

    )



  def forward(self, x):

    x = self.feature_extractor(x)

    x = x.view(-1, 2*2*256)

    x = self.classifier(x)

    return x
# Train Net

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



net = ResNet(models.resnet18(pretrained=True), classes=2)

net = net.to(torch.device(device))



criterion = nn.CrossEntropyLoss(weight=torch.tensor([no_cactus_weight, has_cactus_weight], dtype=torch.float32).to(torch.device(device)))

optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0001)

scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.96)



metrics = {

    'accuracy': {

        'f': accuracy_score,

        'args': {}

    },

    'balanced_accuracy': {

        'f': balanced_accuracy_score,

        'args': {}

    },

    'f1': {

        'f': f1_score,

        'args': {'average': 'weighted'}

    }

}



trainer = PyTorchTrainer(device=device, metrics=metrics)

trainer.train(net, optimizer, criterion, train_dataloader, dev_dataloader, scheduler=scheduler, epochs=100)
# Trigger net to eval mode

net.eval()

for parameter in net.parameters():

  parameter.requires_grad = False
def get_metrics_dict(predictions, targets):

  d = {}

  for metric_name in metrics:

    metric_value = metrics[metric_name]['f'](predictions, targets, **metrics[metric_name]['args'])

    d[metric_name] = metric_value

  return d



# Forward test set

avg_metrics_dict = None

batches = len(test_dataloader)

for batch_data in test_dataloader:

  # Get Inputs and put them to device

  inputs, targets = batch_data

  _inputs = inputs.to(torch.device(device))

  # Forward model to get outputs

  _outputs = net.forward(_inputs)

  # Variables

  predictions = _outputs.argmax(dim=1).data.cpu()

  metrics_dict = get_metrics_dict(predictions, targets)

  # Update variables

  if avg_metrics_dict is None:

    avg_metrics_dict = metrics_dict.copy()

  else:

    for metric_name in avg_metrics_dict:

      avg_metrics_dict[metric_name] += metrics_dict[metric_name]



for metric_name in avg_metrics_dict:

  avg_metrics_dict[metric_name] /= batches

  print(metric_name, avg_metrics_dict[metric_name])
# Submission Dataset Class

class SubmissionDataset(torch.utils.data.Dataset):

  

  def __init__(self, root, transform=None):

    super(SubmissionDataset, self).__init__()

    self.transform = transform

    self.image_filenames = []

    self.image_paths = []

    for dirname, _, filenames in os.walk(root):

      for filename in filenames:

        self.image_filenames.append(filename)

        self.image_paths.append(os.path.join(root, filename))

  

  def __len__(self):

    return len(self.image_filenames)

      

  def __getitem__(self, index):

    image = Image.open(self.image_paths[index])

    return image if self.transform is None else self.transform(image)
# Submission Dataset

submission_dataset = SubmissionDataset(test_data_path, transform=TestTransforms())

# Submission Dataloader

submission_dataloader = torch.utils.data.DataLoader(submission_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# Show submission batch

images = next(iter(submission_dataloader))

fig = make_image_label_figure(images[:9])
# Get all submission predictions

test_predictions = None

for batch_data in submission_dataloader:

  # Get Inputs and put them to device

  inputs = batch_data

  _inputs = inputs.to(torch.device(device))

  # Forward model to get outputs

  _outputs = net.forward(_inputs)

  # Variables

  predictions = _outputs.argmax(dim=1).data

  test_predictions = predictions if test_predictions is None else torch.cat((test_predictions, predictions))

test_predictions = test_predictions.cpu().numpy()
# Write submission to pd.DataFrame

submission_df = pd.DataFrame(np.c_[np.array(submission_dataset.image_filenames)[:,None], test_predictions], columns=['id', 'has_cactus'])
# Write submission DataFrame to csv

submission_df.to_csv('submission.csv', index=False)
!rm -rf data