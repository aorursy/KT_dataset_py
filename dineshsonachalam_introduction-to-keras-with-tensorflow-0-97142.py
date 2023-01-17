# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from __future__ import print_function

from keras.datasets import mnist



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential    # Importing Sequential Model

from keras.layers.core import Dense,Dropout, Activation  #  Importing  Dense Layers,Dropouts and Activation functions

from keras.optimizers import RMSprop, Adam

from keras.utils import np_utils  

np.random.seed(1671) # for reproducibility -> Once you put the same seed you get same patterns of random numbers.





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# create the training & test sets, skipping the header row with [1:]

train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
test= pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

Y_train = train['label'].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32') # All Pixel values
X_train
Y_train
#X_train is 42000 rows of 28x28 values --> reshaped in 60000 x 784



RESHAPED = 784

X_train = X_train.reshape(42000, RESHAPED)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')
# normalize -> Involve only rescaling to arrive at value relative to some size variables.



X_train /= 255 # Pixel values are 0 to 255 -> So we are normalizing training data by dividing it by 255

X_test /= 255

print(X_train.shape[0], 'train samples')

print(X_test.shape[0], 'test samples')
#  Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = np_utils.to_categorical(Y_train, 10) 





# np_utils.to_categorical Used to convert the array of labelled data to one Hot vector-> Binarization of category
# network and training

NB_EPOCH = 30 # 30-> times the model is exposed to the training set.

BATCH_SIZE = 300

VERBOSE = 1

NB_CLASSES = 10 # number of outputs = number of digits

OPTIMIZER = Adam()

N_HIDDEN = 128 # Neurons

VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION

DROPOUT = 0.3
# Final hidden layer  with 10 outputs

# final stage is softmax

model = Sequential() # Sequential Model.

model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,))) # 1st Hidden Layer --> 128 neurons and input dimension ->784

model.add(Activation('relu')) # Activation function for 1st Hidden Layer

model.add(Dropout(DROPOUT))



model.add(Dense(N_HIDDEN))  # 2nd Hidden Layer --> 128 neurons

model.add(Activation('relu')) # Activation function for 2nd Hidden Layer

model.add(Dropout(DROPOUT))





model.add(Dense(NB_CLASSES)) # Final layer with 10 neurons == > no of outputs

model.add(Activation('softmax')) # Final layer activation will be 'softmax'



model.summary()
# Compiling a model in keras

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
# Training a model in keras



# Once the model is compiled it can be trained with the fit() function



history = model.fit(X_train, Y_train,

batch_size=BATCH_SIZE, epochs=NB_EPOCH,

verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({'ImageId':list(range(1,len(predictions) + 1)), "Label": predictions})

submissions.to_csv("Final_Prediction.csv", index=False, header=True)