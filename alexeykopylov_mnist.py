import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Lambda, Flatten

from keras.optimizers import Adam, RMSprop

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
np.random.seed(2)

from random import randint

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

%matplotlib inline
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
test.head()
Y_train = train["label"]

X_train = train.drop("label",axis = 1)

X_test = test
Y_train.value_counts()
X_train.isnull().any().describe()
X_test.isnull().any().describe()
#Normalize the data



# Reshape image in 3 dimensions (height = 28px, width = 28px)



X_train = X_train.values.reshape(-1,28*28)

X_train = (X_train.astype('float32') - 128)/128



X_test = X_test.values.reshape(-1,28*28)

X_test = (X_test.astype('float32') - 128)/128
X_train.shape
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)
# Set the random seed

random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size = 0.1, random_state = random_seed)
#plt.imshow(X_train[randint(0, X_train.shape[0]-1)][:])
#del network
network = Sequential()

network.add(Dense(512, activation = 'relu', input_shape = (28*28,)))

network.add(Dense(10, activation = 'relu'))

network.add(Dense(10, activation = 'relu'))

network.add(Dense(10, activation = 'softmax'))
network.compile(optimizer = 'rmsprop',

               loss = 'categorical_crossentropy',

               metrics = ['accuracy'])
network.fit(X_train,Y_train, epochs = 10, batch_size = 128 )
val_loss, val_acc = network.evaluate(X_val,Y_val)

print("val_acc:", val_acc)
predictions =  network.predict_classes(X_test)
submission = pd.DataFrame({'ImageId':range(1,predictions.shape[0]+1),'Label':predictions})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'MNIST Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)