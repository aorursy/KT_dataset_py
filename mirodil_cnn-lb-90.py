# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from matplotlib import pyplot as plt
from skimage.transform import resize
import torch
from torch.utils.data import TensorDataset, DataLoader

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
path = Path('../input')
train_df = pd.read_json(path/'train.json')
train_df.head()
train_df.dtypes
img = np.array(train_df.audio_embedding[0])
plt.imshow(img)
img = np.array(train_df.audio_embedding[97])
plt.imshow(img)
plt.imshow(resize(img/255, (10,128), mode='reflect'))
plt.imshow(resize(img/255, (10,128), mode='constant'))
def convert_audio_embedding(e):
    img = np.array(e, dtype=np.float32)/255
    if img.shape != (10,128):
        img = resize(img, (10,128), mode='constant')
    return img
plt.imshow(convert_audio_embedding(train_df.audio_embedding[97]))
audio_embeddings = np.array(list(map(convert_audio_embedding,train_df.audio_embedding)))
audio_embeddings = audio_embeddings.reshape((-1,1, 10, 128)).astype(np.float32)
audio_embeddings.shape
is_turkey =train_df.is_turkey.values.reshape((-1,1)).astype(np.float32)
is_turkey.shape
X_tensor = torch.from_numpy(audio_embeddings)
y_tensor = torch.from_numpy(is_turkey)
train_dataset = TensorDataset(X_tensor, y_tensor)
dl = DataLoader(train_dataset, batch_size=16,shuffle=True)
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 16, 2)
        self.fc1 = nn.Linear(16 * 1 * 31, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 1 * 31)
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


net = Net()
x, y = iter(dl).next()
y_hat = net(x)
x.shape,y.shape, y_hat.shape
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.)
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dl, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('[{0:d},{1:2d}] loss: {2:.3f}'.format(epoch + 1, i + 1, running_loss / len(dl)))
print('Finished Training')
x, y = iter(dl).next()
y_hat = net(x)
y-(y_hat>0.5).float()
test_df = pd.read_json(path/'test.json')
test_df.shape
audio_embeddings = np.array(list(map(convert_audio_embedding,test_df.audio_embedding)))
audio_embeddings = audio_embeddings.reshape((-1,1, 10, 128)).astype(np.float32)
audio_embeddings.shape
X_tensor = torch.from_numpy(audio_embeddings)
pred = net(X_tensor)
pred = (pred>0.6).int().numpy()
submission = pd.DataFrame({'vid_id': test_df.vid_id.values, 'is_turkey': pred.flatten()})
submission.head(5)
submission.to_csv("submission.csv", index=False)
