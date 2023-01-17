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
data_all = pd.read_csv('../input/train.csv')
data = data_all.drop(columns = ['label'])
from keras.utils import np_utils

data_labels = np_utils.to_categorical(np.array(data_all['label']), 10) 
data = data.values
import tensorflow as tf
import tflearn

# Define the neural network


def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    # Inputs
    net = tflearn.input_data([None, data.shape[1]])

    # Hidden layer(s)
    net = tflearn.fully_connected(net, 128, activation='ReLU')
    net = tflearn.fully_connected(net, 32, activation='ReLU')
    
    # Output layer and training model
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')
    
    model = tflearn.DNN(net)
    return model

model = build_model()
model.fit(data, data_labels, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=20)
data_test = pd.read_csv('../input/test.csv')
predictions = np.array(model.predict(data_test.values)).argmax(axis=1)

submission = pd.DataFrame({ 'ImageId': list(range(1,len(predictions)+1)), 'Label': predictions})
submission.to_csv('submission.csv', index=False)

