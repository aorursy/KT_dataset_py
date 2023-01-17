import pandas as pd

import numpy as np



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.autograd import Variable

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from torchvision.utils import make_grid



import math, random, numbers

from PIL import Image
INPUT_DIR = '../input/digit-recognizer'

BATCH_SIZE = 64

N_EPOCHS = 50
train_df = pd.read_csv(INPUT_DIR + '/train.csv')



n_train = len(train_df)

n_pixels = len(train_df.columns) - 1

n_class = len(set(train_df['label']))



print('Number of training samples: {0}'.format(n_train))

print('Number of training pixels: {0}'.format(n_pixels))

print('Number of classes: {0}'.format(n_class))
test_df = pd.read_csv(INPUT_DIR + '/test.csv')



n_test = len(test_df)

n_pixels = len(test_df.columns)



print('Number of train samples: {0}'.format(n_test))

print('Number of test pixels: {0}'.format(n_pixels))
class MNIST_data(Dataset):

    def __init__(self, file_path, 

                 transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 

                     transforms.Normalize(mean=(0.5,), std=(0.5,))])

                ):

        

        df = pd.read_csv(file_path)

        

        if len(df.columns) == n_pixels:

            # test data

            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = None

        else:

            # training data

            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = torch.from_numpy(df.iloc[:,0].values)

            

        self.transform = transform

    

    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        if self.y is not None:

            return self.transform(self.X[idx]), self.y[idx]

        else:

            return self.transform(self.X[idx])
class RandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None):

        if isinstance(degrees, numbers.Number):

            if degrees < 0:

                raise ValueError("If degrees is a single number, it must be positive.")

            self.degrees = (-degrees, degrees)

        else:

            if len(degrees) != 2:

                raise ValueError("If degrees is a sequence, it must be of len 2.")

            self.degrees = degrees



        self.resample = resample

        self.expand = expand

        self.center = center



    @staticmethod

    def get_params(degrees):

        angle = np.random.uniform(degrees[0], degrees[1])

        return angle



    def __call__(self, img):

        def rotate(img, angle, resample=False, expand=False, center=None):

            return img.rotate(angle, resample, expand, center)

        angle = self.get_params(self.degrees)

        return rotate(img, angle, self.resample, self.expand, self.center)
class RandomShift(object):

    def __init__(self, shift):

        self.shift = shift

        

    @staticmethod

    def get_params(shift):

        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift 

    

    def __call__(self, img):

        hshift, vshift = self.get_params(self.shift)        

        return img.transform(img.size, Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC, fill=1)
train_dataset = MNIST_data(INPUT_DIR + '/train.csv', transform=transforms.Compose(

                            [transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(3),

                             transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))

test_dataset = MNIST_data(INPUT_DIR + '/test.csv')



train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class Net(nn.Module):    

    def __init__(self):

        super(Net, self).__init__()

          

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)

        )

          

        self.classifier = nn.Sequential(

            nn.Dropout(p = 0.5),

            nn.Linear(64 * 7 * 7, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(p = 0.5),

            nn.Linear(512, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(p = 0.5),

            nn.Linear(512, 10),

        )

          

        for m in self.features.children():

            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()

        

        for m in self.classifier.children():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, nn.BatchNorm1d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()



    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x
model = Net()



optimizer = optim.Adam(model.parameters(), lr=0.003)

criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()
def train(epoch):

    model.train()

    optimizer.step()

    exp_lr_scheduler.step()



    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data), Variable(target)

        

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

        

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        

        if (batch_idx + 1)% 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))
def evaluate(data_loader):

    model.eval()

    loss = 0

    correct = 0

    

    with torch.no_grad():

        for data, target in data_loader:

            data, target = Variable(data), Variable(target)

            if torch.cuda.is_available():

                data = data.cuda()

                target = target.cuda()



            output = model(data)

            loss += F.cross_entropy(output, target,  reduction='sum').item()

            pred = output.data.max(1, keepdim=True)[1]

            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        

    loss /= len(data_loader.dataset)

        

    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(

        loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
for epoch in range(N_EPOCHS):

    train(epoch)

    evaluate(train_loader)
def prediciton(data_loader):

    model.eval()

    test_pred = torch.LongTensor()

    

    with torch.no_grad():

        for i, data in enumerate(data_loader):

            data = Variable(data)

            if torch.cuda.is_available():

                data = data.cuda()



            output = model(data)

            pred = output.cpu().data.max(1, keepdim=True)[1]

            test_pred = torch.cat((test_pred, pred), dim=0)

        

    return test_pred
test_pred = prediciton(test_loader)
out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset)+1)[:,None], test_pred.numpy()], columns=['ImageId', 'Label'])
out_df.head()
out_df.to_csv('submission.csv', index=False)