import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision.transforms as transforms

import torchvision

import numbers

from PIL import Image, ImageOps, ImageEnhance

import os

import matplotlib.pyplot as plt

%matplotlib inline
if torch.cuda.is_available():

    current_device = torch.cuda.current_device()

    torch.cuda.set_device(current_device)

    device = torch.device('cuda:{}'.format(current_device))

    print('Using GPU: {}'.format(torch.cuda.get_device_name(current_device)))

else:

    device = torch.device('cpu')
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
class MNIST(torch.utils.data.Dataset):    

    def __init__(self, file_path, train=False,

                 transform = transforms.Compose(

                     [transforms.ToPILImage(),transforms.ToTensor(),

                      transforms.Normalize(mean=(0.5,),std=(0.5,))])

                ):

        df = pd.read_csv(file_path)

        

        if train:

             # training data

            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = torch.from_numpy(df.iloc[:,0].values)

        else:

             # test data

            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = None

            

        self.transform = transform

    

    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        if self.y is not None:

            return self.transform(self.X[idx]), self.y[idx]

        else:

            return self.transform(self.X[idx])
class Net(nn.Module):

    

    def __init__(self):

        super().__init__()

        

        # feature encoders/identifiers

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))



        # classifier MLP

        self.classifier = nn.Sequential(

            nn.Dropout(p=0.2),

            nn.Linear(32 * 7 * 7, 784),

            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(784, 784),

            nn.ReLU(),

            nn.Linear(784, 10))



    def forward(self, X):

        x = self.features(X)

        x = x.view(x.size(0), 32 * 7 * 7)

        x = self.classifier(x)

        return x



net = Net()

print(net)
n_epochs = 105

learning_rate = 1e-3

batch_size = 1024

w_decay = 2e-4



# use 80% for training, 20% for validation

data_split = 0.8

rnd_shift = 3

rnd_rotation = 20



model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)

criterion = nn.CrossEntropyLoss()
fulltrainset = MNIST('/kaggle/input/digit-recognizer/train.csv', train=True,

                     transform =transforms.Compose([transforms.ToPILImage(),

                                                    RandomRotation(degrees=rnd_rotation),

                                                    RandomShift(rnd_shift),

                                                    transforms.ToTensor(),

                                                    transforms.Normalize(mean=(0.5,), std=(0.5,))

                                                   ])

                    )

# Split dataset into validation and training set

train_size = int(data_split * len(fulltrainset))

test_size = len(fulltrainset) - train_size

trainset, validationset = torch.utils.data.random_split(fulltrainset, [train_size, test_size])



trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)



validloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=2)



# Test set

testset = MNIST('/kaggle/input/digit-recognizer/test.csv', train=False)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
def validate(dataloader):

    model.eval()

    correct = 0

    losses = []

    

    for data, target in dataloader:

        with torch.no_grad():

            data, target = torch.autograd.Variable(data).to(device), torch.autograd.Variable(target).to(device)

            output = model(data)

            loss = F.cross_entropy(output, target, reduction='mean')

            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()

            losses.append(loss.item())

    

    acc = 100. * correct / len(dataloader.dataset)

    print('\nValidation loss: {:.4f}\tAccuracy: {}/{} ({:.0f}%)\n'.format(

        np.mean(np.array(losses)), correct, len(dataloader.dataset), acc))

    

    return np.mean(np.array(losses)), acc
def train(dataloader, epoch):

    model.train()

    losses = []

    

    for batch_idx, (data, target) in enumerate(dataloader):

        data, target = torch.autograd.Variable(data).to(device), torch.autograd.Variable(target).to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

        

        if (batch_idx + 1) % 10 == 0:

            print('Train Epoch: {} [{}/{}]\tLoss: {:.4f}'.format(

                epoch, ((batch_idx + 1) * len(data)), (len(dataloader.dataset)), np.mean(np.array(losses))))



    return np.mean(np.array(losses))
train_losses = []

test_losses = []

accuracies = []



for epoch in range(n_epochs):

    loss = train(trainloader, epoch)

    test_loss, acc = validate(validloader)

    

    train_losses.append(loss)

    test_losses.append(test_loss)

    accuracies.append(acc)
fig,ax = plt.subplots(2,1,figsize=(14,18))



ax[0].plot(test_losses, label='Validation loss')

ax[0].plot(train_losses, label='Training loss')

ax[0].set_xlabel('Epoch')

ax[0].set_ylabel('Loss')

ax[0].legend(loc='upper right')



ax[1].plot(accuracies)

ax[1].set_xlabel('Epoch')

ax[1].set_ylabel('Accuracy')
model.eval()

test_pred = torch.LongTensor()



for i, data in enumerate(testloader):

    with torch.no_grad():

        data = torch.autograd.Variable(data)

        data = data.to(device)

        output = model(data)

        pred = output.cpu().data.max(1, keepdim=True)[1]

        test_pred = torch.cat((test_pred, pred), dim=0)

        

out_df = pd.DataFrame(np.c_[np.arange(1, len(testset)+1)[:,None], test_pred.numpy()], 

                      columns=['ImageId', 'Label'])

out_df.head()

out_df.to_csv('submission.csv', index=False)