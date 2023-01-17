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
PATH = '/kaggle/input/digit-recognizer/'
import torch 

import torchvision.transforms as transforms

import torch.utils.data



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
def show_image(pixels, ax=plt):

    ax.imshow(pixels.reshape(28, 28), cmap='gray')
train_df = pd.read_csv(PATH + 'train.csv')

test_df = pd.read_csv(PATH + 'test.csv')

combined_df = train_df.append(test_df, sort=False)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_df.drop('label', axis=1), train_df['label'], test_size=0.2, random_state=17)
X_train = X_train.values.reshape(len(X_train), 28, 28)

X_val = X_val.values.reshape(len(X_val), 28, 28)



X_train = X_train / 255

X_val = X_val / 255
train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))

test_ds = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))



train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
fig, axs = plt.subplots(1, 5, figsize=(20, 12))

i = 0

for n in np.random.choice(len(train_df), 5):

    show_image(np.array(train_df.drop('label', axis=1).iloc[n]), ax=axs[i])

    axs[i].axis('off')

    i += 1
fig = plt.figure(figsize=(10, 6))

digits_frequency = train_df['label'].value_counts()

sns.barplot(list(range(0, 10)), digits_frequency)
import torch.nn as nn

import torch.nn.functional as F



class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)

        

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        

        return x



cnn = CNN()
import torch.optim as optim

from torch.autograd import Variable



critertion = nn.CrossEntropyLoss()

optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
for epoch in range(15):  # loop over the dataset multiple times

    

    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader, 0):

        inputs = Variable(images.view(64, 1, 28, 28))

        labels = Variable(torch.tensor(list(map(int, labels))))



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = cnn(inputs)

        loss = critertion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 100 == 99:    # print every 100 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 100))

            running_loss = 0.0



print('Finished Training')
from sklearn.metrics import confusion_matrix, f1_score



pred = cnn(torch.tensor(X_val, dtype=torch.float32).view(len(X_val), 1, 28, 28))

pred = torch.max(pred.data, 1)[1]



print('Confusion mastrix:\n', confusion_matrix(y_val, pred))

print('F1 score: ', f1_score(y_val, pred, average='macro'))
X_test = test_df

X_test = X_test.values.reshape(len(X_test), 28, 28)

X_test = X_test / 255
sub = cnn(torch.tensor(X_test, dtype=torch.float32).view(len(X_test), 1, 28, 28))

sub = pd.DataFrame({

                'ImageId': range(1, len(sub.data) + 1),

                'Label': torch.max(sub.data, 1)[1].numpy()

            }, columns=['ImageId', 'Label'])

print(sub.head())

sub.to_csv('submission.csv', index=False)