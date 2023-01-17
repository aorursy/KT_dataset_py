!pip install albumentations
import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision.transforms as transforms

import albumentations

from albumentations import torch as AT

import torchvision.models as models

import cv2



from torch.utils.data import Dataset

from torch.autograd import Variable

from typing import Optional

from typing import Tuple

from typing import List

from PIL import Image

import pandas as pd

import numpy as np

import math

import random



import time

import datetime

import IPython.display as display

from IPython.display import clear_output



import matplotlib.pyplot as plt

import matplotlib.animation as animation

from matplotlib import style

import pylab as pl



from google.colab import files


DATA_FOLDER = "../input"
LR = 0.1

BATCH_SIZE = 256

TEST_BATCH_SIZE = 512

WEIGHT_DECAY = 0.0005



CLASSES = ['Cancer', 'No cancer']

  

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_float_tensor(tensor):

    if not torch.is_tensor(tensor):

        tensor = torch.FloatTensor(tensor)

    else:

        tensor = tensor.type(torch.FloatTensor)

    return tensor.cuda()



class MainDataset(Dataset):

    def __init__(self,

                 x_dataset: Dataset,

                 y_dataset: Dataset,):

        self.x_dataset = x_dataset

        self.y_dataset = y_dataset



    def __len__(self) -> int:

        return self.x_dataset.__len__()



    def __getitem__(self, index: int) -> Tuple:

        x = self.x_dataset[index]

        y = self.y_dataset[index]

        return x, y





class ImageDataset(Dataset):

    def __init__(self, paths_to_imgs: List,

                 transformer: Optional = None):

        self.paths_to_imgs = paths_to_imgs

        self.transformer = transformer



    def __len__(self) -> int:

        return len(self.paths_to_imgs)



    def __getitem__(self, index: int) -> Image.Image:

        img = cv2.imread(self.paths_to_imgs[index])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transformer is not None:

            img = self.transformer(image=img)

        return img['image']





class LabelDataset(Dataset):

    def __init__(self, labels: List):

        self.labels = labels



    def __len__(self) -> int:

        return len(self.labels)



    def __getitem__(self, index: int) -> int:

        return self.labels[index]





class IdDataset(Dataset):

    def __init__(self, ids: List):

        self.ids = ids



    def __len__(self) -> str:

        return len(self.ids)



    def __getitem__(self, index: int) -> str:

        return self.ids[index]
kaggle_train_data = pd.read_csv(f"{DATA_FOLDER}/train_labels.csv")



valid_data = kaggle_train_data.sample(n=2500, random_state=113)

kaggle_train_data = kaggle_train_data.drop(valid_data.index.values)



test_data = kaggle_train_data.sample(n=2500, random_state=113)

kaggle_train_data = kaggle_train_data.drop(test_data.index.values)



train_data = kaggle_train_data
transformers = {

    'train': albumentations.Compose([

    albumentations.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),

    albumentations.RandomRotate90(p=0.5),

    albumentations.Transpose(p=0.5),

    albumentations.Flip(p=0.5),

    albumentations.OneOf([

        albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(), 

        albumentations.RandomBrightness(), albumentations.RandomContrast(),

        albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5), 

    albumentations.HueSaturationValue(p=0.5), 

    albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),

    albumentations.Normalize(),

    AT.ToTensor()

    ]),

    'test': albumentations.Compose([

    albumentations.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),

    albumentations.Normalize(),

    AT.ToTensor()

    ])

}


train_labels = pd.read_csv(f"{DATA_FOLDER}/train_data.csv")['label'].values.reshape(-1, 1)

valid_labels = pd.read_csv(f"{DATA_FOLDER}/valid_data.csv")['label'].values.reshape(-1, 1)

test_labels = pd.read_csv(f"{DATA_FOLDER}/test_data.csv")['label'].values.reshape(-1, 1)

submission_labels = pd.read_csv(f"{DATA_FOLDER}/sample_submission.csv")['label'].values.reshape(-1, 1)



train_ids = pd.read_csv(f"{DATA_FOLDER}/train_data.csv")['id'].values.reshape(-1, 1)

valid_ids = pd.read_csv(f"{DATA_FOLDER}/valid_data.csv")['id'].values.reshape(-1, 1)

test_ids = pd.read_csv(f"{DATA_FOLDER}/test_data.csv")['id'].values.reshape(-1, 1)

submission_ids = pd.read_csv(f"{DATA_FOLDER}/sample_submission.csv")['id'].values.reshape(-1, 1)



train_images = [f"{DATA_FOLDER}/train/{f[0]}.tif" for f in train_ids]

valid_images = [f"{DATA_FOLDER}/train/{f[0]}.tif" for f in valid_ids]

test_images = [f"{DATA_FOLDER}/train/{f[0]}.tif" for f in test_ids]

submission_images = [f"{DATA_FOLDER}/test/{f[0]}.tif" for f in submission_ids]



train_labels_dataset = LabelDataset(train_labels)

valid_labels_dataset = LabelDataset(valid_labels)

test_labels_dataset = LabelDataset(test_labels)

submission_labels_dataset = LabelDataset(submission_labels)



train_images_dataset = ImageDataset(train_images, transformers['train'])

valid_images_dataset = ImageDataset(valid_images, transformers['test'])

test_images_dataset = ImageDataset(test_images, transformers['test'])

submission_images_dataset = ImageDataset(submission_images, transformers['test'])



train_dataset = MainDataset(train_images_dataset, train_labels_dataset)

valid_dataset = MainDataset(valid_images_dataset, valid_labels_dataset)

test_dataset = MainDataset(test_images_dataset, test_labels_dataset)

submission_dataset = MainDataset(submission_images_dataset, submission_labels_dataset)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

submission_loader = torch.utils.data.DataLoader(submission_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)
fig = plt.figure(figsize=(20, 4))

for idx, img in enumerate(np.random.choice(train_images, 10)):

    ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])

    im = Image.open(img)

    plt.imshow(im)

    lab = train_labels[np.where(train_ids == img.split('/')[-1].split('.')[0])][0]

    ax.set_title(f'Label: {CLASSES[lab]}')
fig = plt.figure(figsize=(20, 4))

for idx, img in enumerate(np.random.choice(test_images, 10)):

    ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])

    im = Image.open(img)

    plt.imshow(im)

    lab = test_labels[np.where(test_ids == img.split('/')[-1].split('.')[0])][0]

    ax.set_title(f'Label: {CLASSES[lab]}')
def positive_labels(data_loader, samples_num=None):

    if samples_num is None:

        return sum([torch.sum(labels).item() for i, (images, labels) in enumerate(data_loader, 0)])

    else:

        positive_sum = 0

        for i, (images, labels) in enumerate(train_loader, 0):

            positive_sum += torch.sum(labels).item()

            if i > samples_num:

              break

        return positive_sum



def total_labels(data_loader):

    return len(data_loader) * data_loader.batch_size

  

if SHOW_POSITIVES:

    valid_positives = positive_labels(valid_loader)

    test_positives = positive_labels(test_loader)

    train_positives = positive_labels(train_loader, 30)



    valid_total = total_labels(valid_loader)

    test_total = total_labels(test_loader)

    train_total = train_loader.batch_size * 30



    print('Positives:total')

    print(f'Valid dataset: {valid_positives}:{valid_total}')

    print(f'Test dataset: {test_positives}:{test_total}')

    print(f'Train dataset: {train_positives}:{train_total}')

def set_x_max(x1, x2):

    x_max = 200

    if x1.size != 0:

        x_max = max(x1[-1] + 100, x_max)

    if x2.size != 0:

        x_max = max(x2[-1] + 100, x_max)

    return x_max

        



def init_plot(title, x1, y1, x2, y2):

    clear_output()

    plt.style.use('seaborn-whitegrid')

    plt.xlabel('Mini-batch', fontsize=15)

    plt.ylabel('Cost', fontsize=15)

    plt.xlim(0, set_x_max(x1, x2))

    plt.ylim(0, 1)

    plt.scatter(x1, y1, label='Train', marker='o', color='blue')

    plt.scatter(x2, y2, label='Validation', marker='o', color='red')

    plt.legend()

    fig = plt.gcf()

    fig.suptitle(title, fontsize=30)

    fig.set_size_inches(20,10)

    display(fig)





def update_plot(x1, y1, x2, y2):

    plt.xlim(0, set_x_max(x1, x2))

    plt.scatter(x1, y1, label='Costs', marker='o', color='blue')

    plt.scatter(x2, y2, label='Errors', marker='o', color='red')

    clear_output(wait=True)

    display(plt.gcf())
def test_error(model, data_loader):

    correct = 0

    total = 0

    with torch.no_grad():

        for (inputs, labels) in data_loader:

            inputs, labels = to_float_tensor(inputs), to_float_tensor(labels)

            outputs = model(inputs)

            predicted = torch.round(torch.sigmoid(outputs))

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        return 1 - correct / total



def test_cost(model, data_loader):

    running_loss = 0.0

    mini_batches = 0

    with torch.no_grad():

        for (inputs, labels) in data_loader:

            inputs, labels = to_float_tensor(inputs), to_float_tensor(labels)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            mini_batches += 1

        return running_loss / mini_batches



def outputs_for(model, data_loader, ids):

    loader_len = len(data_loader)

    result = {'id': np.array([]), 'label': np.array([]), 'output': np.array([])}

    print_interval = 100

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(data_loader, 0):

            inputs = to_float_tensor(inputs)

            result['label'] = np.append(result['label'], labels.data.cpu().numpy().reshape(-1))

            result['output'] = np.append(result['output'], torch.sigmoid(model(inputs)).data.cpu().numpy().reshape(-1))

            if i % print_interval == print_interval - 1:

                print(f'{datetime.datetime.now().replace(microsecond=0)} mini-batch: {(i + 1):3d}/{loader_len}')

    result['id'] = ids.reshape(-1)

    return result



def calculate_roc(test_result, step):

    roc = {'x': np.array([]), 'y': np.array([])}

    for threshold in np.arange(0, 1 + step, step):

        roc['x'] = np.append(roc['x'], threshold)

        predictions = test_result['output'] >= threshold

        results = [prediction == label for prediction, label in zip(predictions, test_result['label'])]

        roc['y'] = np.append(roc['y'], np.sum(results) / len(results))

    return roc



def draw_roc(roc):

    plt.plot(roc['x'], roc['y'], color='black', linewidth='3')

    plt.xlim(0, 1)

    plt.ylim(0, 1)

    plt.show()



def calculate_auroc(roc):

    return np.sum([(roc['x'][i + 1] - roc['x'][i])*(roc['y'][i] + 0.5 * (roc['y'][i + 1] - roc['y'][i])) for i in range(len(roc['x']) - 1)])



def eval_submission(submission, threshold):

    result = {'id': submission['id'], 'label': []}

    result['label'] = (submission['output'] >= threshold).astype(int)

    return result

    
learning_start_time = time.time()

train_costs = (np.array([]), np.array([]))

valid_costs = (np.array([]), np.array([]))

epoch = 0

batch_num = 0

avg_train_loss = 0.7
%pylab inline



def train_model(model):

    global train_costs, valid_costs

    init_plot('Training model ' + str(TRAINED_MODEL), train_costs[0], train_costs[1], valid_costs[0], valid_costs[1])

    print(f'{datetime.datetime.now().replace(microsecond=0)} Started learning')



    global learning_start_time

    global batch_num

    global avg_train_loss

    global epoch



    for i in range(EPOCHS):

        running_loss = 0.0

        start_time = time.time()

        dataset_len = len(train_loader.dataset)



        for i, (inputs, labels) in enumerate(train_loader, 0):

            inputs, labels = to_float_tensor(inputs), to_float_tensor(labels)



            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()



            running_loss += loss.item()

            print_interval = 150

            valid_interval = 400

            if i % print_interval == print_interval - 1:

                avg_train_loss = avg_train_loss * 0.7 + (running_loss / print_interval) * 0.3

                running_loss = 0.0

                train_costs = (np.append(train_costs[0], batch_num),

                               np.append(train_costs[1], avg_train_loss))

                update_plot(train_costs[0], train_costs[1], valid_costs[0], valid_costs[1])

                print(f'[{datetime.datetime.now().replace(microsecond=0)} epoch: {epoch + 1}, mini-batch: {(i + 1):3d}/{math.ceil(dataset_len / BATCH_SIZE)}] loss: {avg_train_loss:.3f} in: {time.time() - start_time:.0f} s')

                start_time = time.time()

            if i % valid_interval == valid_interval - 1:

                valid_cost = test_cost(model, valid_loader)

                valid_costs = (np.append(valid_costs[0], batch_num),

                               np.append(valid_costs[1], valid_cost))

                update_plot(train_costs[0], train_costs[1], valid_costs[0], valid_costs[1])

                print(f"{datetime.datetime.now().replace(microsecond=0)} Loss on validation dataset: {valid_cost:1.3f}")

                print(f"{datetime.datetime.now().replace(microsecond=0)} Total learning time: {((time.time() - learning_start_time) / 60):3.1f} min")

                start_time = time.time()

            batch_num += 1

        epoch += 1
class ResUnit(nn.Module):



    def __init__(self, inplanes, outplanes):

        super(ResUnit, self).__init__()

        self.inplanes = inplanes

        self.outplanes = outplanes

        self.downsample = nn.Conv2d(inplanes, outplanes, 1)

        self.conv = nn.Sequential(

            nn.Conv2d(inplanes, int(outplanes / 4), 1),

            nn.BatchNorm2d(int(outplanes / 4)),

            nn.ReLU(inplace=True),

            nn.Conv2d(int(outplanes / 4), int(outplanes / 4), 3, padding=1),

            nn.BatchNorm2d(int(outplanes / 4)),

            nn.ReLU(inplace=True),

            nn.Conv2d(int(outplanes / 4), outplanes, 1),

            nn.BatchNorm2d(outplanes)

        )

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):

        residual = x

        x = self.conv(x)

        if self.inplanes != self.outplanes:

            residual = self.downsample(residual)

        x += residual

        return self.relu(x)

class ResNet(nn.Module):



    def __init__(self):

        super(ResNet, self).__init__()

        # 68 x 68

        self.conv1 = nn.Sequential(

            nn.Conv2d(3, 64, 5),

            nn.ReLU(inplace=True)

        )

        # 32 x 32

        self.conv2 = nn.Sequential(

            ResUnit(64, 256),

            ResUnit(256, 256),

            ResUnit(256, 256)

        )

        # 16 x 16

        self.conv3 = nn.Sequential(

            ResUnit(256, 512),

            ResUnit(512, 512),

            ResUnit(512, 512),

            ResUnit(512, 512)

        )

        # 8 x 8

        self.conv4 = nn.Sequential(

            ResUnit(512, 1024),

            ResUnit(1024, 1024),

            ResUnit(1024, 1024),

            ResUnit(1024, 1024),

            ResUnit(1024, 1024),

            ResUnit(1024, 1024)

        )

        # 4 x 4

        self.conv5 = nn.Sequential(

            ResUnit(1024, 2048),

            ResUnit(2048, 2048),

            ResUnit(2048, 2048)

        )

        self.max_pool = nn.MaxPool2d(2, 2)

        self.avg_pool = nn.AvgPool2d(4, 1)

        # 2048

        self.fc = nn.Linear(2048, 1)



    def forward(self, x):

        x = self.conv1(x)

        x = self.max_pool(x)

        x = self.conv2(x)

        x = self.max_pool(x)

        x = self.conv3(x)

        x = self.max_pool(x)

        x = self.conv4(x)

        x = self.max_pool(x)

        x = self.conv5(x)

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

model1 = ResNet()

model1 = model1.cuda()
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
train_model(model1)
error = test_error(model1, valid_loader)  

print(f"Error on valid dataseti: {(error * 100):2.2f}%")
error = test_error(model1, test_loader)  

print(f"Error on test dataset: {(error * 100):2.2f}%")
outputs_test = outputs_for(model1, train_loader, test_ids)

roc = calculate_roc(outputs_test, 0.001)

auroc = calculate_auroc(roc)

argmax = np.argmax(roc['y'])

clear_output(wait=True)

draw_roc(roc)

print(f'AUROC: {auroc:.3f}')

print(f"Max of ROC: {roc['y'][argmax]:.3f} for threshold: {roc['x'][argmax]}")
submission = outputs_for(model1, submission_loader, submission_ids)

submission_result = eval_submission(submission, 0.5)

pd.DataFrame.from_dict(submission_result).to_csv('submission.csv', index=False)