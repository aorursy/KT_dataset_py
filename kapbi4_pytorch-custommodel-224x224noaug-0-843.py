#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



#/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv

#/kaggle/input/plant-pathology-2020-fgvc7/test.csv

#/kaggle/input/plant-pathology-2020-fgvc7/train.csv

#/kaggle/input/plant-pathology-2020-fgvc7/images/Test_1283.jpg

#/kaggle/input/plant-pathology-2020-fgvc7/images/Train_1334.jpg
import torch

from torch import nn, optim

from torch.nn import Module as M

from torch.utils.data import Dataset as D

# import torchvision

from torchvision import datasets # for mnist

import torchvision.transforms as transforms



import matplotlib.pyplot as plt

import matplotlib.cm as cm

import pandas as pd

import os

import numpy as np

import random

import math

import tqdm

#from tqdm import tqdm



import albumentations as A

from albumentations.pytorch import ToTensorV2 as AT



import cv2



#from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold



import logging

IMG_SIZE = 224

epoch_count = 10

#patience = 0

fold_count = 5

SEED = 42



batch_size = 64 * 2



# Скорость обучения

LR = 5e-5



# Параметры оптимизатора Adam

#beta1 = 0.5

beta1 = 0.9

beta2 = 0.999



BASE_PATH = '/kaggle/input/plant-pathology-2020-fgvc7'

#BASE_PATH = './data'

train_file = os.path.join(BASE_PATH, "train.csv")

test_file = os.path.join(BASE_PATH, "test.csv")

submission_file = os.path.join(BASE_PATH, "sample_submission.csv")



logging.basicConfig(filename='./strem_v1_(224x224noAug).log',

                    format='%(asctime)s - %(levelname)s - %(message)s',

                    level=logging.INFO)

logging.info('________________________________________________________')

logging.info('Запуск скрипта')

logging.info('fold_count:{} epoch_count:{} batch_size:{} LR:{}'.format(fold_count, epoch_count, batch_size, LR))
train_df = pd.read_csv(train_file)

train_df.head()
test_df = pd.read_csv(test_file)

test_df.head()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cpu')
submission_df = pd.read_csv(submission_file)

submission_df.iloc[:, 1:] = 0



submission_df.head()
class digitModel(M):

    def __init__(self):

        super(digitModel, self).__init__()

        # расчета размера выходного слоя после Conv2d

        # c_out = ((c_in+2pading-kernel_size)/strides)+1

        #

        self.cnn01 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) # 224

        self.bn01 = nn.BatchNorm2d(16)

        self.relu01 = nn.ReLU()

        self.maxpool01 = nn.MaxPool2d(kernel_size=2)  # 112

        #

        self.cnn02 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 112

        self.bn02 = nn.BatchNorm2d(32)

        self.relu02 = nn.ReLU()

        self.maxpool02 = nn.MaxPool2d(kernel_size=2)  # 56

        #

        self.cnn03 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 56

        self.bn03 = nn.BatchNorm2d(64)

        self.relu03 = nn.ReLU()

        self.maxpool03 = nn.MaxPool2d(kernel_size=2)  # 28

        # compress chanel 128 to 3

        #self.cnn04 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0) # 28

        #self.bn04 = nn.BatchNorm2d(3)

        #self.relu04 = nn.ReLU()

        # Convolution 1

        self.cnn1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 28

        self.bn1 = nn.BatchNorm2d(128)

        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 14

        # Convolution 2

        self.cnn2 = nn.Conv2d(in_channels=128, out_channels=254, kernel_size=3, stride=1, padding=0) # 12

        self.bn2 = nn.BatchNorm2d(254)

        self.relu2 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # 6

        # Convolution 3

        self.cnn3 = nn.Conv2d(in_channels=254, out_channels=512, kernel_size=3, stride=1, padding=0) # 4

        self.bn3 = nn.BatchNorm2d(512)

        self.relu3 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # 2

        # Convolution 4

        self.cnn4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1) # 2

        self.bn4 = nn.BatchNorm2d(1024)

        self.relu4 = nn.ReLU()

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)  # 1

        # Fully connected 1

        self.fc1 = nn.Linear(1024 * 1 * 1, 4)



    def forward(self, x):

        ##

        x = self.cnn01(x)

        x = self.bn01(x)

        x = self.relu01(x)

        x = self.maxpool01(x)

        ##

        x = self.cnn02(x)

        x = self.bn02(x)

        x = self.relu02(x)

        x = self.maxpool02(x)

        ##

        x = self.cnn03(x)

        x = self.bn03(x)

        x = self.relu03(x)

        x = self.maxpool03(x)

        ## Compress

        #x = self.cnn04(x)

        #x = self.bn04(x)

        #x = self.relu04(x)

        # Convolution 1

        x = self.cnn1(x)

        x = self.bn1(x)

        x = self.relu1(x)

        x = self.maxpool1(x)

        # Convolution 2

        x = self.cnn2(x)

        x = self.bn2(x)

        x = self.relu2(x)

        x = self.maxpool2(x)

        # Convolution 3

        x = self.cnn3(x)

        x = self.bn3(x)

        x = self.relu3(x)

        x = self.maxpool3(x)

        # Convolution 4

        x = self.cnn4(x)

        x = self.bn4(x)

        x = self.relu4(x)

        x = self.maxpool4(x)

        # подготовка для линейного слоя

        x = x.view(x.size(0), -1)

        # Linear function (readout)

        x = self.fc1(x)

        return x
class digitDataset(D):

    def __init__(self, df, transform=None): #, labels=None

        if 'healthy' in df: # ['healthy','multiple_diseases','rust','scab']

            self.images_id = df['image_id'].values

            self.labels = df.drop(axis=1, columns='image_id').values

            self.labels = self.labels[:, 1] + self.labels[:, 2] * 2 + self.labels[:, 3] * 3

        else:

            self.images_id = df['image_id'].values

            self.labels = np.zeros(len(df))

        

        self.transform = transform

            

    def __len__(self):

        return len(self.images_id)

    

    def __getitem__(self, idx):

        image_src = BASE_PATH + '/images/' + self.images_id[idx] + '.jpg'

        image = cv2.imread(image_src) #, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = self.labels[idx]

        

        if self.transform:

            transformed = self.transform(image=image)

            image = transformed['image']



        image = np.multiply(image, 1/255)

        image = torch.from_numpy(np.array(image, dtype=np.float32))



        return image, labels

def imshow(imgs, lbls, epoh='', batch=''):

    x = int(math.sqrt(len(imgs)))+1

    y = x + 1

    fig = plt.figure(figsize=(15, 16))

    for i, img in enumerate(imgs):

        img = np.rollaxis(img.numpy(), 0, 3)

        img = np.uint8(img)

        ax = fig.add_subplot(x, y, i+1)

        ax.set_title(str(lbls.numpy()[i]))

        ax.imshow(img, cmap = cm.binary)

        plt.xticks(np.array([]))

        plt.yticks(np.array([]))

    plt.show()
logging.info('Назначение модели')

model = digitModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR, betas=(beta1, beta2))

criterion = nn.CrossEntropyLoss()
def train(model, train_loader, criterion, optimizer, show=False, show_first=False):

    model.train()

    tr_loss = 0

    

    for step, batch in enumerate(tqdm.tqdm(train_loader)):

        images = batch[0]

        labels = batch[1]

        

        if show:

            imshow(images, labels) # ,transform)

        if show_first and step == 0:

            imshow(images, labels) # ,transform)

        

        images = images.to(device) # , dtype=torch.float)

        labels = labels.to(device)

                    

        outputs = model(images)

        loss = criterion(outputs, labels) #.squeeze(-1))                

        loss.backward()



        tr_loss += loss.item()



        optimizer.step()

        optimizer.zero_grad()

        

    return tr_loss / len(train_loader)
def valid(model, valid_loader, criterion, optimizer):

    model.eval()

    val_loss = 0

    correct = 0

    count = 0



    for step, batch in enumerate(tqdm.tqdm(valid_loader)):



            images = batch[0]

            labels = batch[1]



            count += len(images)



            images = images.to(device)

            labels = labels.to(device)



            with torch.no_grad():

                outputs = model(images)



                loss = criterion(outputs, labels)

                val_loss += loss.item()



                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum().item()

                    

    return val_loss / len(valid_loader), correct / count
def test(model, test_loader, criterion, optimizer):

    model.eval()

    test_preds = None

    

    for step, batch in enumerate(tqdm.tqdm(test_loader)):



        images = batch[0]

        images = images.to(device) #, dtype=torch.float)



        with torch.no_grad():

            outputs = model(images)



            

            if test_preds is None:

                test_preds = outputs.data.cpu()

            else:

                test_preds = torch.cat((test_preds, outputs.data.cpu()), dim=0)

    return test_preds

train_transform = A.Compose([

    #A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),

    #A.Flip(),

    #A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),

    A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),

    #A.Normalize(p=1.0),

    AT(p=1.0),

])



test_transform = A.Compose([

    A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),

    #A.Normalize(p=1.0),

    AT(p=1.0),

])
logging.info('Запуск тренировки')

folds = KFold(n_splits=fold_count, shuffle=True, random_state=SEED)



show_first = False

for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df)):



    train_data = train_df.loc[train_idx]

    valid_data = train_df.loc[valid_idx]



    #Инициализируем датасеты



    trainset = digitDataset(train_data, transform=train_transform)

    validset = digitDataset(valid_data, transform=test_transform)



    train_loader = torch.utils.data.DataLoader(trainset, 

                                           batch_size = batch_size, 

                                           shuffle = True)

    valid_loader = torch.utils.data.DataLoader(validset, 

                                          batch_size = batch_size, 

                                          shuffle = False)

    

    for epoch in range(epoch_count):

        logging.info('Запуск эпохи тренировки')

        tr_loss = train(model, train_loader, criterion, optimizer, show_first=show_first)

        logging.info('Запуск эпохи валидации')

        val_loss, predicted = valid(model, valid_loader, criterion, optimizer)

        logging.info('fold:{} epoch:{} tr_loss:{:.4f} val_loss:{:.4f}, acc:{:.4f}'.format(i_fold+1, epoch+1, tr_loss, val_loss, predicted))

        print('fold:{} epoch:{} tr_loss:{:.4f} val_loss:{:.4f}, acc:{:.4f}'.format(i_fold+1, epoch+1, tr_loss, val_loss, predicted))

        show_first = False

        

        torch.save({'model_state_dict': model.state_dict(),

                    'optimizer_state_dict': optimizer.state_dict(),

                    'epoch': epoch,

                    'loss': tr_loss,

                    },

                   'model.pth.tar',

                   )

logging.info('Запуск тестирования')

test_data = pd.read_csv(test_file)





testset = digitDataset(test_data, transform=test_transform)



test_loader = torch.utils.data.DataLoader(testset, 

                                          batch_size = batch_size, 

                                          shuffle = False)



test_preds = test(model, test_loader, criterion, optimizer)



submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_preds, dim=1)

submission_df.to_csv('submission.csv', index=False)

logging.info('Завершение скрипта')