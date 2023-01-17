import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



import matplotlib.pyplot as plt
import numpy as np

import torch

import torch.nn as nn

import torch.cuda as tc

from torch.autograd import Variable

import torch.nn.functional as F
dt = pd.read_pickle('../input/cifar-100/train')

df = pd.DataFrame.from_dict(dt, orient='index')

df = df.T

df.head()
labels = list(pd.read_pickle('../input/cifar-100/meta')['fine_label_names'])
# dictionaries with labels to idx and viceversa

labels_idx = {v:k for (k,v) in enumerate(labels)}

idx_labels = {k:v for (k,v) in enumerate(labels)}
# display a bee image

img = np.array(df[df['fine_labels'] == labels_idx['bee']].iloc[10]['data']).reshape(3,32,32)

plt.imshow(img.transpose(1,2,0).astype("uint8"), interpolation='nearest')
data_values = df[df['fine_labels'].isin([labels_idx['bee'], labels_idx['butterfly']])]['data'].values

label_values = df[df['fine_labels'].isin([labels_idx['bee'], labels_idx['butterfly']])]['fine_labels'].values
features = np.array(np.stack(data_values, axis=0)).reshape(len(data_values), 3, 32, 32).astype('float32')

labels = np.array(label_values).astype('float32')
# normalize

features = features / 255.0



features = torch.from_numpy(features)

labels = torch.from_numpy(labels)
# create one hot enconding

labels[labels == labels_idx['bee']] = 0

labels[labels == labels_idx['butterfly']] = 1

labels = torch.eye(2)[labels.type(torch.LongTensor)]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
torch.manual_seed(999)



model = nn.Sequential(

            nn.Conv2d(3, 16, 3),

            nn.Dropout(0.2),

            nn.MaxPool2d(3),

            nn.Flatten(),

            nn.Linear(1600, 1024),

            nn.Tanh(),

            nn.Linear(1024, 512),

            nn.Tanh(),

            nn.Linear(512, 256),

            nn.Tanh(),

            nn.Linear(256, 2),

            nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)



# kfold split

kf = KFold(n_splits=3)
cnt = 1

losses = []

losses_test = []



for x_idx in kf.split(X_train):

    for e in range(501):

        for idx in x_idx:

            optimizer.zero_grad()

            out = model(X_train[idx])

            loss = criterion(out, torch.argmax(torch.Tensor(y_train[idx]).type(torch.torch.LongTensor), dim=1))

            loss.backward()

            optimizer.step()

        if(e % 100 == 0):

            pred_test = model(X_test)

            loss_test = criterion(pred_test, torch.argmax(torch.Tensor(y_test).type(torch.torch.LongTensor), dim=1))

            

            losses.append(loss)

            losses_test.append(loss_test)

            

            print('Split:%3d, Epoch:%4d, Loss:%.3f, Loss-test:%.3f' % (cnt, e, loss.item(), loss_test.item()))

    cnt += 1
fig, ax = plt.subplots()

ax.plot(losses, label='Losses')

ax.plot(losses_test, label='Losses-Test')

leg = ax.legend()