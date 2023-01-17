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
from keras.models import Sequential

from keras.optimizers import SGD

from keras.utils import np_utils

from matplotlib.pyplot import imshow



########################

# Import the layer types needed

from keras.layers.core import Activation, Dense, Dropout, Flatten

from keras.layers.convolutional import Convolution2D,MaxPooling2D

########################
num_classes = 10



training_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



y_train = training_data['label'].values

X_train = training_data.drop('label', 1).values

X_test = test_data.values



X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))

X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

y_train = np_utils.to_categorical(y_train, num_classes)


#################################

# Design the model here

model = Sequential()



# 2d convolutional layer with 32 filters of size 3*3

model.add(Convolution2D(nb_filter=32,nb_row=3,nb_col=3,input_shape=(1, X_train.shape[2], X_train.shape[3]), dim_ordering='th'))



# ReLU nonlinearity

model.add(Activation("relu"))



# max pooling layer

model.add(MaxPooling2D(pool_size=(2, 2)))



# 2D convolutional layer with 64 filters of size 3*3

model.add( Convolution2D( nb_filter = 64, nb_row=3, nb_col=3 ) )



# ReLU nonlinearity

model.add( Activation("relu") )



# max pooling layer

model.add(MaxPooling2D(pool_size=(2, 2)))



# flatten layer

model.add(Flatten())



# fully connected layer with 128 neurons

model.add(Dense(output_dim = 128))



# dropout layer with drop probability 0.5

model.add(Dropout(0.5))



# ully-connected layer with 10 neurons

model.add(Dense(output_dim = 10))



# softmax layer for summing probability up to 1

model.add(Activation("softmax"))





##################################
# Now compile and fit the model



model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))



## Fit the model (10% of training data used as validation set)

model.fit(X_train, y_train, nb_epoch=7, batch_size=32,validation_split=0.1, show_accuracy=True, verbose=0)
y_predict = model.predict(X_test)



y_predict_values = np.argmax(y_predict, axis=1)
predicted_data = pd.DataFrame(data=y_predict_values, columns=['label'])

predicted_data.index.name = 'ImageId'



predicted_data.to_csv(path_or_buf='Answer.csv')