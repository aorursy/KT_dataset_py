# packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



# Time to say good-bye, chainer...

import chainer

import chainer.links as L

import chainer.functions as F
# get datas

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

print(test_df.shape)

print(train_df.shape)
# load dataframe to numpy

X_train, Y_train, X_test, Y_test= [], [], [], []

X_train = train_df.iloc[:,1:].values.reshape(-1,1,28,28)

Y_train = train_df.iloc[:,0].values



print(len(X_train))

print(len(Y_train))
# Transfer 32

X_train = X_train.astype('float32')

Y_train = Y_train.astype('int32')
# Dataset for Chainer

dataset = chainer.datasets.TupleDataset(X_train, Y_train)

len(dataset)
# 70% as training

n_train = int(len(dataset) * 0.7)

train, test = chainer.datasets.split_dataset_random(dataset, n_train, seed=1)



# check raws

len(train)
# very simple CNN

class CNN(chainer.Chain):



    def __init__(self, n_mid=100, n_out=10):

        super().__init__()

        with self.init_scope():

            self.conv1 = L.Convolution2D(in_channels=1, out_channels=3, ksize=6, stride=1, pad=1)

            self.conv2 = L.Convolution2D(in_channels=1, out_channels=3, ksize=3, stride=1, pad=1)

            self.fc1 = L.Linear(None, n_mid)

            self.fc2 = L.Linear(None, n_out)



    def __call__(self, x):

        h = F.relu(self.conv1(x))

        h = F.max_pooling_2d(h, 3, 3)

        h = F.relu(self.conv2(x))

        h = F.max_pooling_2d(h, 3, 3)

        h = self.fc1(h)

        h = self.fc2(h)

        return h
# define random seed

import random

def reset_seed(seed=0):

    random.seed(seed)

    np.random.seed(seed)

    if chainer.cuda.available:

        chainer.cuda.cupy.random.seed(seed)
reset_seed(0)

model = L.Classifier(CNN())

gpu_id = -1

# none gpu

# when gpu use

# gpu_id = 0

# model.to_gpu(gpu_id)
# define optimizer, and set up with model

optimizer = chainer.optimizers.Adam()

optimizer.setup(model)



batchsize = 96

train_iter = chainer.iterators.SerialIterator(train, batchsize)

test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=True)



from chainer import training

from chainer.training import extensions



# set epoch

epoch = 10

updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

trainer = training.Trainer(updater, (epoch, 'epoch'), out='mnist')



# validate with 30% train data

trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))



# logging learning process

trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))

# Print Report per 1 poch

trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy', 'main/loss', 'validation/main/loss', 'elapsed_time']), trigger=(1, 'epoch'))
# start training

trainer.run()
import json

with open('mnist/log') as f:

    result = pd.DataFrame(json.load(f))

result
# 損失関数(loss)

result[['main/loss', 'validation/main/loss']].plot()
# 精度(accuracy)

result[['main/accuracy', 'validation/main/accuracy']].plot()
X_test = test_df.iloc[:,1:].values.reshape(-1,1,28,28)

print(X_test.shape)

#X_test = test_df.values.reshape(1,28,28)

X_test = X_test.astype('float32')
# predict with test data



def predict(model, x_dataset):

    y = model.predictor(x_dataset)

    return np.argmax(y.data, axis = 1)



y_test = predict(model, X_test)

y_test
df_y_test = pd.DataFrame({'label':y_test})

df_Y_id = test_df.iloc[:,0]

df_submit = pd.concat([df_Y_id,df_y_test],axis=1)

df_submit.columns = ['id', 'label']

df_submit
df_submit.to_csv('submission.csv', index=False)