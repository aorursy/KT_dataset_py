# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
len(train['ID'])
import theano as T
from random import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv1DLayer),
        ('pool1', layers.MaxPool1DLayer),
        ('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 369),
    conv1_num_filters=32, conv1_filter_size=10, pool1_pool_size=2,
    #hidden4_num_units=500,
    #dropout1_p=0.1,
    #conv2_num_filters=64, conv2_filter_size=(1,3), pool2_pool_size=(1,2),
    #hidden5_num_units=500,
    dropout2_p=0.1,
    output_num_units=1, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=100,
    verbose=1,
    )
X=np.array(train[train.columns[1:-1]].values)
y=train[train.columns[-1]].values
X = X.astype(np.float32)
y = y.astype(np.float32)
z=X.reshape(76020, 369)
z.shape
T.tensor3(z)
net2.fit(X
,y)
