!git clone https://github.com/jakubkala/deep-learning-methods.git

!mv deep-learning-methods/* .
import os

import numpy as np

import pandas as pd

import random
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
X_train = train.loc[:, list(test.columns)]

X_train = np.array(X_train)

X_test = np.array(test)



y_train = train.label
from data_preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)

X_test = ss.transform(X_test)
from sklearn.preprocessing import OneHotEncoder

from neural_network_wrapper import NeuralNetworkWrapper

import optimizers
y_ohc = np.zeros((y_train.size, int(np.max(y_train))+1))

y_ohc[np.arange(y_train.size), y_train.astype(np.int)] = 1

y_train = y_ohc
d = {

    "input_dim" : X_train.shape[1],

    "neuron_numbers" : [128, 128, 128, 128, 10], 

    "activation_functions" : ['relu', 'relu', 'relu', 'relu', 'softmax'],

    "loss_function" : 'max_likelihood_loss', #do we need this?

    "batch_size" : 128,

    "num_epochs" : 100,

    "learning_rate" : 0.001,

    "beta": 0.97

}
NN = NeuralNetworkWrapper(d['input_dim'],

                          d['neuron_numbers'],

                          d['activation_functions'],

                          d['loss_function'],

                          d['learning_rate'],

                          optimizers.GDwithMomentum(d['beta']),

                          d['batch_size'])
NN.train(X_train,

         y_train,

         d['num_epochs'],

         validation_split = 0.1,

         verbosity=True)
y_pred = NN.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
submission=pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),

                         "Label": y_pred})

submission.to_csv("submission.csv", index=False, header=True)
NN.plot_loss()
submission.head()
for i in range(10):

    print(np.sum(y_pred==i) / len(y_pred))