import torch

import torch.nn as nn

import torch.optim as optim

import torchvision.transforms as transforms

from torch.utils.data import Dataset, SubsetRandomSampler

from torch.utils.tensorboard import SummaryWriter

import pandas as pd

import numpy as np

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

from datetime import datetime, date



torch.manual_seed(0)

np.random.seed(0)



# TRAIN_PATH = 'train.csv'

# TEST_PATH = 'test.csv'

# SUBMISSION_PATH = 'submission.csv'

# DIG_PATH = 'Dig-MNIST.csv'

TRAIN_PATH = '../input/Kannada-MNIST/train.csv'

TEST_PATH = '../input/Kannada-MNIST/test.csv'

SUBMISSION_PATH = '/kaggle/working/submission.csv'

DIG_PATH = '../input/Kannada-MNIST/Dig-MNIST.csv'
# !kaggle competitions download -c Kannada-MNIST

# !unzip Kannada-MNIST.zip
class KannadaMNISTDataset(Dataset):

    def __init__(self, csv_path, transform):

        data = pd.read_csv(csv_path).to_numpy()

        self.y = data[:, 0]

        self.X = data[:, 1:].reshape(-1, 28, 28).astype(np.uint8)

        assert self.y.shape[0] == self.X.shape[0]

        self.transform = transform



    def __len__(self):

        return self.X.shape[0]



    def __getitem__(self, idx):

        return self.transform(self.X[idx]), self.y[idx]

    

    def asPIL(self, idx):

        return transforms.ToPILImage()(self.X[idx]), self.y[idx]

    

    def merge(self, other):

        self.X = np.concatenate((self.X, other.X))

        self.y = np.concatenate((self.y, other.y))
transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((32, 32)),

    transforms.RandomAffine(degrees=10, translate=(0.25, 0.25), scale=(0.75, 1.25), shear=0.1),

    transforms.ToTensor()

])

dataset = KannadaMNISTDataset(TRAIN_PATH, transform)

dataset.merge(KannadaMNISTDataset(DIG_PATH, transform))



train_idx, test_idx = train_test_split(range(len(dataset)), train_size=0.85)

train_sampler = SubsetRandomSampler(train_idx)

test_sampler = SubsetRandomSampler(test_idx)



BATCH_SIZE = 512

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=8, pin_memory=True)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=8, pin_memory=True)
class SimpleNet(nn.Module):

    def __init__(self):

        super(SimpleNet, self).__init__()

        def make_conv_group(in_, out_):

            return nn.Sequential(

                nn.Conv2d(in_, out_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

                nn.BatchNorm2d(out_, momentum=0.05),

                nn.ReLU(inplace=True)

            )

        

        self.layers = nn.Sequential(

            make_conv_group(1, 64),

            make_conv_group(64, 128),

            make_conv_group(128, 128),

            make_conv_group(128, 128),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout2d(p=0.1),

            make_conv_group(128, 128),

            make_conv_group(128, 128),

            make_conv_group(128, 256),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout2d(p=0.1),

            make_conv_group(256, 256),

            make_conv_group(256, 256),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout2d(p=0.1),

            make_conv_group(256, 512),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout2d(p=0.1),

            make_conv_group(512, 2048),

            make_conv_group(2048, 256),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout2d(p=0.1),

            make_conv_group(256, 256)

        )

        self.head = nn.Linear(256, 10)

        

        for layer in self.layers.modules():

          if isinstance(layer, nn.Conv2d):

            nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

            

    def forward(self, x):

        x = self.layers(x)

        x = x.view(x.size(0), -1)

        x = self.head(x)

        x = nn.Softmax(dim=1)(x)

        return x
net = SimpleNet().cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=1e-5, verbose=True)

EPOCH = 0
for epoch in tqdm(range(EPOCH, EPOCH + 100), leave=False):

    net.train()

    for batch_x, batch_y in tqdm(train_loader, leave=False):

        optimizer.zero_grad()



        outputs = net(batch_x.cuda())

        loss = criterion(outputs, batch_y.cuda())

        loss.backward()

        optimizer.step()

  

    test_loss = 0.0

    net.eval()

    for batch_x, batch_y in tqdm(test_loader, leave=False):

        outputs = net(batch_x.cuda())

        loss = criterion(outputs, batch_y.cuda())



        test_loss += loss.item()

    test_loss /= len(test_idx)

    

    scheduler.step(test_loss)
test_dataset = KannadaMNISTDataset(TEST_PATH, transform)

loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

test_answers = None

net.eval()

for batch_x, _ in tqdm(loader):

    outputs = net(batch_x.cuda()).cpu().detach().numpy()

    if test_answers is not None:

        test_answers = np.concatenate((test_answers, outputs))

    else:

        test_answers = outputs

good_ind = test_answers.max(axis=1) >= 0.95

good_labels = test_answers[good_ind].argmax(axis=1)
pseudo_dataset = KannadaMNISTDataset(TEST_PATH, transform)

pseudo_dataset.X = pseudo_dataset.X[good_ind]

pseudo_dataset.y = good_labels

pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)



for lr in [1e-4, 1e-5, 1e-6]:

    for g in optimizer.param_groups:

        g['lr'] = lr

    for epoch in tqdm(range(EPOCH, EPOCH + 100), leave=False):

        net.train()

        for batch_x, batch_y in tqdm(pseudo_loader, leave=False):

            optimizer.zero_grad()



            outputs = net(batch_x.cuda())

            loss = criterion(outputs, batch_y.cuda())

            loss.backward()

            optimizer.step()



        EPOCH += 1
test_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((32, 32)),

    transforms.ToTensor()

])

test_dataset = KannadaMNISTDataset(TEST_PATH, test_transform)

loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)



results = []

net.eval()

for batch_x, _ in tqdm(loader):

    outputs = net(batch_x.cuda())

    results.extend(map(lambda x: x.item(), outputs.argmax(dim=1)))



submission = pd.DataFrame({

    'id': range(len(test_dataset)),

    'label': results

})

submission.to_csv(SUBMISSION_PATH, index=False)