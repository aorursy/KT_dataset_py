# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

print(df_train.shape, '\n', df_train.head())

df_test = pd.read_csv('../input/test.csv')

print(df_test.shape, '\n', df_test.head())
from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms
class MDataset(Dataset):

    def __init__(self, input_file, train=True, 

                 transforms=transforms.Compose([transforms.ToTensor()])):

        df = pd.read_csv(input_file)

        self.transforms = transforms

        self.train = train

        if train:

            self.y = df['label'].values

            df = df.drop(columns = 'label')

        self.x = df.values.reshape((-1,28,28,1)).astype(np.uint8)

        

    def __len__(self):

        return self.x.shape[0]

    def __getitem__(self, idx):

        if self.train:

            return self.transforms(self.x[idx]), self.y[idx]

        else:

            return self.transforms(self.x[idx])

train_dataset = MDataset('../input/train.csv', train=True)

im,lbl = train_dataset[0]

print(im.shape, im.dtype, lbl)
class DRModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))



        self.classifier = nn.Sequential(

            nn.Dropout(p=0.2),

            nn.Linear(32 * 7 * 7, 784),

            nn.ReLU(),

            nn.Linear(784, num_classes))



    def forward(self, X):

        x = self.features(X)

        x = x.view(x.size(0), 32 * 7 * 7)

        x = self.classifier(x)

        return x
batch_size = 64



train_dataset = MDataset('../input/train.csv', train=True)



indices = np.random.permutation(len(train_dataset))

train_examples = int(0.9 * len(train_dataset))

train_sampler = SubsetRandomSampler(indices[:train_examples])

cv_sampler = SubsetRandomSampler(indices[train_examples:])



train_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                           batch_size=batch_size,

                                          sampler=train_sampler)

cv_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                       batch_size=batch_size,

                                       sampler=cv_sampler)

im,lbl = train_dataset[0]

print(lbl)
if torch.cuda.is_available():

    print('Using CUDA')

    device = torch.device('cuda')

else:

    print('CUDA not available. Using CPU')

    device = torch.device('cpu')

model = DRModel(num_classes=10)

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),

                                 lr=1e-4,

                                 weight_decay=.003)

losses_scores = []

for epoch in range(20):

    epoch_losses = []

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        logits = model(images)

        loss = loss_fn(logits, labels)



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        epoch_losses.append(loss.item())



    epoch_scores = []



    for images, labels in cv_loader:



        images = images.to(device)

        labels = labels.to(device)



        logits = model(images)

        _, labels_pred = logits.max(dim=1)

        score = (labels_pred == labels).float().mean()

        epoch_scores.append(score.item())



    losses_scores.append({'epoch': epoch,

                          'loss': epoch_losses,

                          'score': epoch_scores})



    if epoch % 5 == 0 or epoch == 30 - 1:

        print(f'epoch={epoch:g}, '

              f'loss={np.mean(epoch_losses):g}, '

              f'cv_score={np.mean(epoch_scores):g}, '

              f'cv_score_std={np.std(epoch_scores):g}')



import matplotlib.pyplot as plt

plt.figure(figsize=(11, 7))



for metric in ['loss', 'score']:

    (pd.concat({d['epoch']: pd.Series(d[metric], name=metric)

                for d in losses_scores},

               names=['epoch'])

     .groupby('epoch').mean()

     .plot(label=metric))



plt.axhline(0, ls='--')

plt.axhline(1, ls='--')

plt.legend();
test_dataset = MDataset('../input/test.csv', train=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,

                                           batch_size=batch_size, shuffle=False)

model.eval()

test_pred = torch.LongTensor()



for i, data in enumerate(test_loader):

#    data = Variable(data, volatile=True)

    if torch.cuda.is_available():

        data = data.cuda()



    output = model(data)



    pred = output.cpu().data.max(1, keepdim=True)[1]

    test_pred = torch.cat((test_pred, pred), dim=0)
test_pred.numpy()[:4]
out_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset)+1)[:,None], test_pred.numpy()], 

                      columns=['ImageId', 'Label'])

out_df.head()
a= np.arange(len(test_pred)) + 1

b = np.squeeze(test_pred.numpy())

print(a.shape, b.shape, a.dtype, b.dtype)
out_df1 = pd.DataFrame({'ImageId': np.arange(len(test_pred)) + 1, 'Label': np.squeeze(test_pred.numpy())})

out_df1.head()

out_df.to_csv('submission.csv', index=False)