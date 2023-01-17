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
import numpy as np

import pandas as pd



dataset = pd.read_csv('../input/train.csv')

target = dataset[[0]].values.ravel()

train = dataset.iloc[:,1:].values

test = pd.read_csv('../input/test.csv')



print(dataset.head())
# convert to array and reshape

train = np.array(train).reshape(-1,1,28,28)

test = np.array(test).reshape(-1,1,28,28)



# plot

import matplotlib.pyplot as plt

import matplotlib.cm as cm



plt.imshow(train[10000][0],cmap=cm.binary)
import lasagne

from lasagne import layers

from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet

from nolearn.lasagne import visualize



net1 = NeuralNet(

        layers=[('input', layers.InputLayer),

                ('hidden', layers.DenseLayer),

                ('output', layers.DenseLayer),

                ],

        # layer parameters:

        input_shape=(None,1,28,28),

        hidden_num_units=1000, # number of units in 'hidden' layer

        output_nonlinearity=lasagne.nonlinearities.softmax,

        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9



        # optimization method:

        update=nesterov_momentum,

        update_learning_rate=0.0001,

        update_momentum=0.9,



        max_epochs=15,

        verbose=1,

        )



net1.fit(train,target)

pred = net1.predict(test)



# save results

np.savetxt('submission_nn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

def CNN(n_epochs):

    net1 = NeuralNet(

        layers=[

        ('input', layers.InputLayer),

        ('conv1', layers.Conv2DLayer),      #Convolutional layer.  Params defined below

        ('pool1', layers.MaxPool2DLayer),   # Like downsampling, for execution speed

        ('conv2', layers.Conv2DLayer),

        ('hidden3', layers.DenseLayer),

        ('output', layers.DenseLayer),

        ],



    input_shape=(None, 1, 28, 28),

    conv1_num_filters=7, 

    conv1_filter_size=(3, 3), 

    conv1_nonlinearity=lasagne.nonlinearities.rectify,

        

    pool1_pool_size=(2, 2),

        

    conv2_num_filters=12, 

    conv2_filter_size=(2, 2),    

    conv2_nonlinearity=lasagne.nonlinearities.rectify,

        

    hidden3_num_units=1000,

    output_num_units=10, 

    output_nonlinearity=lasagne.nonlinearities.softmax,



    update_learning_rate=0.0001,

    update_momentum=0.9,



    max_epochs=n_epochs,

    verbose=1,

    )

    return net1



cnn = CNN(15).fit(train,target) # train the CNN model for 15 epochs



# use the NN model to classify test data

pred = cnn.predict(test)



# save results

np.savetxt('submission_cnn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
