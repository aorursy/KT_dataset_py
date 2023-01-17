# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

from torch import nn

from torch.utils import data

from torchvision import models, utils, transforms

from PIL import Image

import zipfile

import time

import copy
to_extract = ['train', 'test1']

for file in to_extract:

    with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/' + file + '.zip', 'r') as z:

        z.extractall('./data')  # 解压train.zip的所有文件到./data目录

    

print('train', len(os.listdir('./data/train')), os.listdir('./data/train')[0])

print('test', len(os.listdir('./data/test1')), os.listdir('./data/test1')[0])
for i in range(5):

    img = Image.open('./data/train/cat.' + str(i) + '.jpg')

    print(img.size)
class DogCat(data.Dataset):

    """

    数据集类，可利用index获取一条数据

    """

    def __init__(self, root, train=True, test=False):

        self.test = test

        imgs = os.listdir(root)

        

        if self.test:

            self.imgs = sorted(imgs, key=lambda x: int(x.split('.')[0]))

        else:

            # 对图片排序，方便按比例分割数据集

            imgs = sorted(imgs, key=lambda x: int(x.split('.')[1]))

            length = len(imgs)

            if train:  # 训练集

                self.imgs = imgs[:int(0.8*length)]

            else:  # 验证集

                self.imgs = imgs[int(0.8*length):]

        self.imgs = [os.path.join(root, img) for img in self.imgs]

        

        if train:

            self.transforms = transforms.Compose([

                transforms.Resize(256),  # 按比例缩放，最小边size=256

                transforms.RandomCrop(224),  # 随机裁剪固定尺寸

                transforms.RandomHorizontalFlip(),  # 随机水平翻转

                transforms.ToTensor(),

                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

            ])

        else:

            self.transforms = transforms.Compose([

                transforms.Resize(256),  # 按比例缩放，最小边size=256

                transforms.CenterCrop(224),  # 中心裁剪固定尺寸

                transforms.ToTensor(),

                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

            ])

    

    def __getitem__(self, index):

        """

        一次返回一张图片的数据，此时加载数据节约内存

        """

        img = Image.open(self.imgs[index])

        img = self.transforms(img)

        

#         print(self.imgs[index])

        if self.test:

            label = int(str(self.imgs[index]).split('/')[-1].split('.')[0])

        else:

            label = 1 if 'dog' in str(self.imgs[index]).split('/')[-1] else 0

        return img, label

    

    def __len__(self):

        return len(self.imgs)
dataset = DogCat('./data/train')

print(dataset[0][1])

# dataset = DogCat('./data/test1', train=False, test=True)

# print(dataset[0][1])
import matplotlib.pyplot as plt



fig, axes = plt.subplots(3, figsize=(6, 12))

for i in range(3):

    img = dataset[i][0].detach().numpy().transpose((1, 2, 0))

    img = img * 0.5 + 0.5 

    axes[i].imshow(img)

plt.show()
class MyNet(nn.Module):

    def __init__(self):

        super(MyNet, self).__init__()

        self.feature = nn.Sequential(

            nn.Conv2d(3, 16, 3, 1, 1),  # (3, 224, 224) -> (16, 224, 224)

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),   # (16, 224, 224) -> (16, 112, 112)

            nn.Conv2d(16, 32, 3, 1, 1),  # (16, 112, 112) -> (32, 112, 112)

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),   # (32, 112, 112) -> (32, 56, 56)

            nn.Conv2d(32, 64, 3, 1, 1),  # (32, 56, 56) -> (64, 56, 56)

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),   # (64, 56, 56) -> (64, 28, 28)

            nn.Dropout(0.5),

            nn.Conv2d(64, 128, 3, 2),  # (128, 28, 28) -> (128, 13, 13)

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),   # (128, 13, 13) -> (128, 6, 6)

            nn.Dropout(0.5)

        )

        

        self.predict = nn.Sequential(

            nn.Linear(128*6*6, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(512, 128), 

            nn.BatchNorm1d(128),

            nn.Dropout(0.25),

            nn.ReLU(inplace=True),

            nn.Linear(128, 2)

        )

        

    def forward(self, x):

        x = self.feature(x)

        x = x.view(x.size(0), -1)

        x = self.predict(x)

        return x

    

model = MyNet()

print(model)

# model(dataset[0][0].clone().detach().resize(1, 3, 224, 224)).size()
torch.cuda.is_available()
class DefaultConfig():

    model = 'MyNet'

    model_path = 'MyNet.pkl'

    result_file = 'submission.csv'

    

    train_root = './data/train'

    test_root = './data/test1'

    

    num_workers = 4

    max_epoch = 30

    batch_size = 128

    lr = 0.001

    device = torch.device('cuda')

    
def train():

    config = DefaultConfig()

    

    # step1: model

    model = eval(config.model)()

    model.to(config.device)

    

    # step2: data

    datasets = {

        'train': DogCat(config.train_root, train=True),

        'val': DogCat(config.train_root, train=False)

    }

    dataloaders = {

        key: data.DataLoader(

            value, 

            batch_size = config.batch_size,

            shuffle=True,

            num_workers=config.num_workers)

        for key, value in datasets.items()

    }

    

    # step3: optimizer and criterion

    criterion = nn.CrossEntropyLoss()

    lr=config.lr

    optimizer = torch.optim.Adam(model.parameters(), lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)



    # step4: train

    since = time.time()

    best_acc = 0.0

    best_model_wts = copy.deepcopy(model.state_dict())

    

    previous_acc = 1e-3

    losses = {'train': [], 'val': []}

    acc = {'train': [], 'val': []}

    for epoch in range(config.max_epoch):

        print('Epoch: {}/{}'.format(epoch, config.max_epoch))

        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':

                lr_scheduler.step()

                model.train()                

            else:

                model.eval()

                

            runing_loss = 0.0

            runing_corrects = 0

            

            with torch.set_grad_enabled(phase == 'train'):  # 训练时才计算梯度

                for step, (img, label) in enumerate(dataloaders[phase]):

                    img = img.to(config.device)

                    label = label.to(config.device)

                    output = model(img)

                    loss = criterion(output, label)



                    if phase == 'train':

                        optimizer.zero_grad()

                        loss.backward()

                        optimizer.step()

                        if step % 50 == 0:

                            print('step: {} | loss: {:.4f}'.format(step, loss.item()))

                   

                    runing_loss += loss.item() * img.size(0)

                    _, pred = torch.max(output.cpu(), 1)

                    runing_corrects += torch.sum(torch.eq(pred, label.cpu())).item()

            

            epoch_loss = runing_loss / len(datasets[phase])

            epoch_acc = runing_corrects / len(datasets[phase])

            print('{} | loss: {:.4f} | Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            losses[phase].append(epoch_loss)

            acc[phase].append(epoch_acc

                             )

#                   

#             if phase == 'val' and epoch_acc < previous_acc:             

#                 # update lr

#                 if lr > 0.00002:

#                     lr *= 0.5

#                     for param_group in optimizer.param_groups:

#                         param_group['lr'] = lr

            

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

            previous_acc = epoch_acc if phase == 'val' else previous_acc

   

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:.4f}'.format(best_acc))

    

    model = model.load_state_dict(best_model_wts)

    torch.save(best_model_wts, config.model_path)

    return model, losses, acc

best_model, losses, acc = train()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(range(len(losses['train'])), losses['train'], color='b', label='train loss')

ax1.plot(range(len(losses['val'])), losses['val'], color='r', label='val loss')

ax1.set_xticks(np.arange(1, 30))

ax1.set_yticks(np.arange(0, 1, 0.1))

plt.legend()



ax2.plot(range(len(acc['train'])), acc['train'], color='b', label='train acc')

ax2.plot(range(len(acc['val'])), acc['val'], color='r', label='val acc')

ax2.set_xticks(np.arange(1, 30))

ax2.set_yticks(np.arange(0, 1.1, 0.1))



plt.legend()

plt.show()
import numpy as np

import pandas as pd



def test():

    config = DefaultConfig()

    

    model = eval(config.model)()

    model.load_state_dict(torch.load(config.model_path))

    model.to(config.device)

    

    test_data = DogCat(config.test_root, train=False, test=True)

    test_loader = data.DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    

    result = []

    for step, (img, path) in enumerate(test_loader):

        img = img.to(config.device)

        path = path.to(config.device)

        output = model(img)



        score, pred = torch.max(nn.functional.softmax(output).cpu(), 1)

        step_result = np.concatenate(

            [path.cpu().view(-1,1).numpy(), 

             pred.detach().cpu().view(-1,1).numpy()], axis=1)

        result.append(step_result)

    result = np.concatenate(result, axis=0)    



    pd.DataFrame(result, columns=['id', 'label']).to_csv(config.result_file, index=False)

test()

df = pd.read_csv('submission.csv')

df
fig, axs = plt.subplots(3, 3, figsize=(12,12))

for i in range(9):

    index = df['id'][i]

    label = df['label'][i]

    

    img = Image.open('./data/test1/' + str(index) + '.jpg')

    axs[i//3, i %3].imshow(np.array(img))

    axs[i//3, i %3].set_title(label)

    

plt.show()