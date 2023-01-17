# Install necessary packages

!git clone https://github.com/NVIDIA/apex  > /dev/null && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ --quiet

!pip install -U git+https://github.com/albu/albumentations > /dev/null
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
root_dir = '../input/traffic-signs-preprocessed/'
import os

import pickle



%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import torch

import cv2

import glob

from apex import amp

from tqdm import tqdm



from albumentations import Compose, ShiftScaleRotate, Resize, Blur, HorizontalFlip, Equalize, Normalize, ElasticTransform

from albumentations.pytorch import ToTensor
label_names = pd.read_csv(root_dir + 'label_names.csv')

label_names.head()
BATCH_SIZE = 128

NUM_CLASSES = 43
train = pickle.load(open(root_dir + 'train.pickle', 'rb'))

valid = pickle.load(open(root_dir + 'valid.pickle', 'rb'))

test = pickle.load(open(root_dir + 'test.pickle', 'rb'))

labels = pickle.load(open(root_dir + 'labels.pickle', 'rb'))
train_images = train['features']

train_labels = train['labels']



valid_images = valid['features']

valid_labels = valid['labels']



test_images = test['features']

test_labels = test['labels']
print(len(train_labels))

print(len(valid_labels))

print(len(test_labels))
fig, axs = plt.subplots(1, 3, figsize=(15,5))



axs[0].hist(train_labels)

axs[0].set_title('Train')



axs[1].hist(valid_labels)

axs[1].set_title('Validation')



axs[2].hist(test_labels)

axs[2].set_title('Test')
MEANS = np.mean(train_images, axis=(0, 1, 2)) / 255.

STDS = np.std(train_images, axis=(0, 1, 2)) / 255.



print(MEANS)

print(STDS)
# Dataset



class TrafficSignsDataset(torch.utils.data.Dataset):



    def __init__(self, images, labels, num_classes, transform=None):

        

        self.images = images

        self.labels = labels

        self.C = num_classes

        self.transform = transform



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        img = self.images[idx]

        label_idx = self.labels[idx]



        label = np.zeros(self.C)

        label[label_idx] = 1

        

        augmented = self.transform(image=img)

        img = augmented['image']

            

        label = torch.tensor(label)

        

        return {'image': img, 'label': label}
# Data loaders



transform_train = Compose([

    Normalize(mean=MEANS, std=STDS),

    Blur(blur_limit=3, p=0.1),

    ShiftScaleRotate(rotate_limit=30, p=0.3),

    ElasticTransform(p=0.1),

    ToTensor()

])



transform_test = Compose([

    Normalize(mean=MEANS, std=STDS),

    ToTensor()

])



train_dataset = TrafficSignsDataset(train_images, train_labels, NUM_CLASSES, transform=transform_train)



valid_dataset = TrafficSignsDataset(valid_images, valid_labels, NUM_CLASSES, transform=transform_test)



test_dataset = TrafficSignsDataset(test_images, test_labels, NUM_CLASSES, transform=transform_test)



data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
# Plot train example



batch = next(iter(data_loader_train))

fig, axs = plt.subplots(1, 5, figsize=(15,5))



print('shape', batch['image'].shape)

print('min', batch['image'].min())

print('max', batch['image'].max())

print('mean', batch['image'].mean())

print('std', batch['image'].std())



for i in np.arange(5):

    img = np.transpose(batch['image'][i].numpy(), (1,2,0))

    img = img * STDS + MEANS

    sign = label_names[label_names.ClassId == torch.argmax(batch['label'][i]).item()]['SignName'].values[0]

    axs[i].imshow(img)

    axs[i].set_title(sign)
# Plot valid example



batch = next(iter(data_loader_valid))

fig, axs = plt.subplots(1, 5, figsize=(15,5))



for i in np.arange(5):

    img = np.transpose(batch['image'][i].numpy(), (1,2,0))

    img = img * STDS + MEANS

    axs[i].imshow(img)

    sign = label_names[label_names.ClassId == torch.argmax(batch['label'][i]).item()]['SignName'].values[0]

    axs[i].set_title(sign)
# Plot test example



batch = next(iter(data_loader_test))

fig, axs = plt.subplots(1, 5, figsize=(15,5))



for i in np.arange(5):

    img = np.transpose(batch['image'][i].numpy(), (1,2,0))

    img = img * STDS + MEANS

    axs[i].imshow(img)

    sign = label_names[label_names.ClassId == torch.argmax(batch['label'][i]).item()]['SignName'].values[0]

    axs[i].set_title(sign)
# Model



class TrafficSignsModel(torch.nn.Module):

    

    def __init__(self, num_classes):

        super().__init__()

        

        self.conv = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),

            torch.nn.ReLU(),

            torch.nn.BatchNorm2d(32),

            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),

            torch.nn.ReLU(),

            torch.nn.BatchNorm2d(32),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),

            torch.nn.ReLU(),

            torch.nn.BatchNorm2d(64),

            torch.nn.Dropout(0.25),

            

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),

            torch.nn.ReLU(),

            torch.nn.BatchNorm2d(64),

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),

            torch.nn.ReLU(),

            torch.nn.BatchNorm2d(64),

            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2),

            torch.nn.ReLU(),

            torch.nn.BatchNorm2d(128),

            torch.nn.Dropout(0.25),

        )

        

        self.flatten = torch.nn.Sequential(torch.nn.AdaptiveMaxPool2d(1), torch.nn.Flatten())

        

        self.fc = torch.nn.Sequential(

            torch.nn.Linear(128, 512),

            torch.nn.ReLU(),

            torch.nn.Dropout(0.3),

            torch.nn.Linear(512, num_classes)

        )

        

    def forward(self, inputs):

        x = self.conv(inputs)

        x = self.flatten(x) #  x.view(-1, 256 * 7 * 7) # 

        x = self.fc(x)

        return x
model = TrafficSignsModel(NUM_CLASSES)



device = torch.device("cuda:0")

model.to(device)



criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)



model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
# Train

n_epochs = 16



for epoch in range(n_epochs):

    

    print('-' * 10)

    print('Epoch {}/{}'.format(epoch + 1, n_epochs))

    print('-' * 10)



    tr_loss = 0

    tr_acc = 0

    model.train()

    for step, batch in enumerate(data_loader_train):



        inputs = batch["image"]

        labels = batch["label"]



        inputs = inputs.to(device, dtype=torch.float)

        labels = labels.to(device, dtype=torch.long)



        outputs = model(inputs)

        loss = criterion(outputs, torch.max(labels, 1)[1])



        with amp.scale_loss(loss, optimizer) as scaled_loss:

            scaled_loss.backward()



        tr_loss += loss.item()

        tr_acc += (torch.max(outputs, 1)[1] == torch.max(labels, 1)[1]).type(torch.FloatTensor).mean().item()



        optimizer.step()

        optimizer.zero_grad()



    epoch_loss = tr_loss / len(data_loader_train)

    epoch_acc = tr_acc / len(data_loader_train)

    print('Training Loss/Accuracy:\t{:.4f}\t{:.4f}'.format(epoch_loss, epoch_acc))

    # print('Training Accuracy: {:.4f}'.format(epoch_acc))

    

    # print('-' * 10)

    test_loss = 0

    test_acc = 0

    model.eval()

    with torch.no_grad():

        for step, batch in enumerate(data_loader_valid):

            inputs = batch["image"]

            labels = batch["label"]



            inputs = inputs.to(device, dtype=torch.float)

            labels = labels.to(device, dtype=torch.long)



            outputs = model(inputs)

            loss = criterion(outputs, torch.max(labels, 1)[1])



            test_loss += loss.item()

            test_acc += (torch.max(outputs, 1)[1] == torch.max(labels, 1)[1]).type(torch.FloatTensor).mean().item()



    epoch_loss = test_loss / len(data_loader_valid)

    epoch_acc = test_acc / len(data_loader_valid)

    print('Validati Loss/Accuracy:\t{:.4f}\t{:.4f}'.format(epoch_loss, epoch_acc))

    # print('Valid Accuracy: {:.4f}'.format(epoch_acc))
errors = []



test_loss = 0

test_acc = 0

model.eval()

with torch.no_grad():

    for step, batch in enumerate(data_loader_test):

        inputs = batch["image"]

        labels = batch["label"]



        inputs = inputs.to(device, dtype=torch.float)

        labels = labels.to(device, dtype=torch.long)



        outputs = model(inputs)

        loss = criterion(outputs, torch.max(labels, 1)[1])



        test_loss += loss.item()

        test_acc += (torch.max(outputs, 1)[1] == torch.max(labels, 1)[1]).type(torch.FloatTensor).mean().item()

        

        true_labels = torch.max(labels, 1)[1]

        pred_labels = torch.max(outputs, 1)[1]

        for idx in range(len(true_labels)):

            if true_labels[idx] != pred_labels[idx]:

                errors.append((np.transpose(inputs[idx].cpu().numpy(), (1,2,0)), true_labels[idx], pred_labels[idx]))



epoch_loss = test_loss / len(data_loader_test)

epoch_acc = test_acc / len(data_loader_test)

print('Test Loss: {:.4f}'.format(epoch_loss))

print('Test Accuracy: {:.4f}'.format(epoch_acc))
print('Error rate', len(errors) / len(test_dataset))
fig, axs = plt.subplots(1, 5, figsize=(15,5))



for i in np.arange(5):

    error = errors[i]

    img = error[0] * STDS + MEANS

    axs[i].imshow(img)

    true_label = label_names[label_names.ClassId == error[1].item()]['SignName'].values[0]

    pred_label = label_names[label_names.ClassId == error[2].item()]['SignName'].values[0]

    axs[i].set_title(true_label + '\n' + pred_label)
!rm -rf /kaggle/working/apex