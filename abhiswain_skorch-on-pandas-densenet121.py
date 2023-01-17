!pip install -U skorch
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2



# torch imports

import torch

from torchvision.transforms import transforms

import torch.optim as optim

import torchvision

import torch.nn as nn

from torch.utils.data import Dataset





# skorch imports

from skorch import NeuralNetClassifier

from skorch.callbacks import LRScheduler, Checkpoint, Freezer

from skorch.helper import predefined_split



import os



# Any results you write to the current directory are saved as output.
%%writefile config.py



train_dir = '../input/pandaresizeddataset512x512/train/'

test_dir = '../input/pandaresizeddataset512x512/test_images/'

train_csv = '../input/prostate-cancer-grade-assessment/train.csv'

test_csv = '../input/prostate-cancer-grade-assessment/test.csv'

batch_size = 32
%%writefile data.py



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import config



# torch imports

import torch

from torch.utils.data import Dataset



class PandaDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.df = pd.read_csv(csv_file)

        self.transform = transform



    def __getitem__(self, index):

        image_ids = self.df['image_id'].values

        labels = self.df['isup_grade'].values



        image = cv2.imread(config.train_dir + image_ids[index] + '.png')

        label = labels[index]



        if self.transform:

            image = self.transform(image)



        image = image.clone().detach()

        label = torch.tensor(label)



        return image, label



    def __len__(self):

        return len(self.df)
%%writefile engine.py



# torch imports

import torch

from torchvision.transforms import transforms

from torch.utils.data import Dataset, random_split



from data import PandaDataset

import config



class Engine:

    def __init__(self):

        self.transforms = transforms.Compose(

            [

                transforms.ToPILImage(),

                transforms.RandomHorizontalFlip(),

                transforms.ToTensor()

            ]

        )

        self.train_loss = []

        self.loss_val = []



    def create_data_loaders(self):

        dataset = PandaDataset(config.train_csv, transform=self.transforms)



        train_size = int(0.8 * len(dataset))

        test_size = len(dataset) - train_size

        train_dataset, valid_dataset = random_split(dataset, [train_size, test_size])



        image_datasets = {

            'train': train_dataset,

            'validation': valid_dataset

        }



        return image_datasets
%%writefile train.py



# torch imports

import torch

import torch.optim as optim

import torchvision

import torch.nn as nn

from torch.optim.lr_scheduler import CyclicLR



# skorch imports

from skorch import NeuralNetClassifier

from skorch.callbacks import LRScheduler, Checkpoint, Freezer

from skorch.helper import predefined_split



from engine import Engine

from data import PandaDataset

import config



class PretrainedModel(nn.Module):

    def __init__(self, output_features):

        super().__init__()

        model = torchvision.models.densenet121(pretrained=True)

        num_ftrs = model.classifier.in_features

        model.classifier = nn.Linear(num_ftrs, output_features)

        self.model = model



    def forward(self, x):

        return self.model(x)



# print(PretrainedModel(6))

# exit(0)

datasets = Engine().create_data_loaders()



lrscheduler = LRScheduler(

    policy='StepLR',

    step_size=7,

    gamma=0.1

)



checkpoint = Checkpoint(

    f_params='densenet_skorch.pt',

    monitor='valid_acc_best'

)



freezer = Freezer(lambda x: not x.startswith('model.classifier'))



net = NeuralNetClassifier(

    PretrainedModel,

    criterion=nn.CrossEntropyLoss,

    batch_size=config.batch_size,

    max_epochs=5,

    module__output_features=6,

    optimizer=optim.SGD,

    iterator_train__shuffle=True,

    iterator_train__num_workers=4,

    iterator_valid__shuffle=True,

    iterator_valid__num_workers=4,

    train_split=predefined_split(datasets['validation']),

    callbacks=[lrscheduler, checkpoint, freezer],

    device='cuda'  # comment to train on cpu

)





#start training

net.fit(datasets['train'], y=None)
!python train.py
