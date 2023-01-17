#Copied from Python CNN notebook

from sklearn.ensemble import RandomForestClassifier

import numpy as np

import pandas as pd



dataset = pd.read_csv("../input/train.csv")

target = dataset[["label"]].values.ravel()

train = dataset.iloc[:,1:].values

test = pd.read_csv("../input/test.csv")



rf = RandomForestClassifier(n_estimators=100)

rf.fit(train,target)

pred = rf.predict(test)

np.savetxt('submission_rand_forest.csv',np.c_[range(1,len(test)+1),pred],delimiter=',',header='ImageId,Label',comments = '',fmt = '%d')

target = target.astype(np.uint8)

train = np.array(train).reshape((-1,1,28,28)).astype(np.uint8)

test = np.array(test).reshape((-1,1,28,28)).astype(np.uint8)

import matplotlib.pyplot as plt

import matplotlib.cm as cm

plt.imshow(train[1729][0],cmap = cm.binary)

import lasagne

from lasagne import layers

from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet

from nolearn.lasagne import visualize



net1 = NeuralNet(

    layers = [

        ('input',layers.InputLayer),

        ('hidden',layers.DenseLayer),

        ('output',layers.DenseLayer),

    ],

    input_shape = (None,1,28,28),

    hidden_num_units = 1000,

    output_nonlinearity = lasagne.nonlinearities.softmax,

    output_num_units = 10,

    update = nesterov_momentum,

    update_learning_rate = 0.0001,

    update_momentum = 0.9,

    max_epochs = 15,

    verbose = 1,

    )

net1.fit(train,target)

print("Done!")
