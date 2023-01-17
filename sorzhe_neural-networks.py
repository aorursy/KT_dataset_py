%matplotlib inline



from sklearn.ensemble import RandomForestClassifier

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score



# create the training & test sets, skipping the header row with [1:]

dataset = pd.read_csv("../input/train.csv")

target = dataset.iloc[0].values.ravel()

train = dataset.iloc[:,1:].values



test = pd.read_csv("../input/test.csv").values
print(dataset.head())
# convert to array, specify data type, and reshape

target = target.astype(np.uint8)

train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)

test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)
import matplotlib.pyplot as plt

import matplotlib.cm as cm



plt.imshow(train[1730][0], cmap=cm.binary) # draw the picture
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



        max_epochs=2,

        verbose=1,

        )
print(net1.fit(train, target))
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
cnn = CNN(5).fit(train,target) # train the CNN model for 15 epochs
# use the NN model to classify test data

pred = cnn.predict(test)



# save results

np.savetxt('submission_cnn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')