import time

import os



import matplotlib.pyplot as plt

import numpy as np



from fastai import *

from fastai.vision import *

import torch

from torchvision import transforms

from sklearn.metrics import classification_report
X_train = np.load('../input/kmnist-train-imgs.npz')['arr_0']

X_test = np.load('../input/kmnist-test-imgs.npz')['arr_0']

y_train = np.load('../input/kmnist-train-labels.npz')['arr_0']

y_test = np.load('../input/kmnist-test-labels.npz')['arr_0']
X_train = X_train.reshape(-1, 1, 28, 28)

X_test = X_test.reshape(-1, 1, 28, 28)
X_train = np.repeat(X_train, 3, axis=1)

X_test = np.repeat(X_test, 3, axis=1)
mean = X_train.mean()

std = X_train.std()

X_train = (X_train-mean)/std

X_test = (X_test-mean)/std



X_train = torch.from_numpy(np.float32(X_train))

y_train = torch.from_numpy(y_train.astype(np.long))

X_test = torch.from_numpy(np.float32(X_test))

y_test = torch.from_numpy(y_test.astype(np.long))
class ArrayDataset(Dataset):

    "Sample numpy array dataset"

    def __init__(self, x, y):

        self.x, self.y = x, y

        self.c = len(np.unique(y))

    

    def __len__(self):

        return len(self.x)

    

    def __getitem__(self, i):

        return self.x[i], self.y[i]
train_ds, valid_ds = ArrayDataset(X_train, y_train), ArrayDataset(X_test, y_test)

data = DataBunch.create(train_ds, valid_ds, bs=64)
learn = cnn_learner(data, models.resnet18, loss_func=CrossEntropyFlat(), metrics=accuracy)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(3, 1e-2)
char_df = pd.read_csv('../input/kmnist_classmap.csv', encoding = 'utf-8')
X,y = learn.get_preds()
print(f"Accuracy of {accuracy(X,y)}")
X = np.argmax(X,axis=1)
target_names = ["Class {} ({}):".format(i, char_df[char_df['index']==i]['char'].item()) for i in range(len(np.unique(y_test)))]

print(classification_report(y, X, target_names=target_names))
X_train = np.load('../input/k49-train-imgs.npz')['arr_0']

X_test = np.load('../input/k49-test-imgs.npz')['arr_0']

y_train = np.load('../input/k49-train-labels.npz')['arr_0']

y_test = np.load('../input/k49-test-labels.npz')['arr_0']
X_train = X_train.reshape(-1, 1, 28, 28)

X_test = X_test.reshape(-1, 1, 28, 28)
X_train = np.repeat(X_train, 3, axis=1)

X_test = np.repeat(X_test, 3, axis=1)
mean = X_train.mean()

std = X_train.std()

X_train = (X_train-mean)/std

X_test = (X_test-mean)/std



# Numpy to Torch Tensor

X_train = torch.from_numpy(np.float32(X_train))

y_train = torch.from_numpy(y_train.astype(np.long))

X_test = torch.from_numpy(np.float32(X_test))

y_test = torch.from_numpy(y_test.astype(np.long))
train_ds, valid_ds = ArrayDataset(X_train, y_train), ArrayDataset(X_test, y_test)

data = DataBunch.create(train_ds, valid_ds, bs=64)
learn = cnn_learner(data, models.resnet18, loss_func=CrossEntropyFlat(), metrics=accuracy)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, 1e-2)
char_df = pd.read_csv('../input/k49_classmap.csv', encoding = 'utf-8')
X,y = learn.get_preds()
print(f"Accuracy of {accuracy(X,y)}")
X = np.argmax(X,axis=1)
target_names = ["Class {} ({}):".format(i, char_df[char_df['index']==i]['char'].item()) for i in range(len(np.unique(y_test)))]

print(classification_report(y, X, target_names=target_names))