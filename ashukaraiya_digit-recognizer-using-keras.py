# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Basic modules for NN
from keras.models import  Sequential 
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import RMSprop
# Modules for CNN

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten, Dropout


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test.head()
Y_train = train["label"]
X_train = train.drop(labels = ["label"], axis = 1)


X_train = X_train.values.reshape(-1, 28, 28 ,1)
test = test.values.reshape(-1,28,28, 1)
X_train = X_train/255.0
test= test/255.0
Y_train = to_categorical(Y_train, num_classes = 10)

# Spliting and validation set
#Set the random Seed
random_seed = 2

# splitting and validating set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1 ,random_state = random_seed)
g = plt.imshow(X_train[0][:,:,0])

def convolutional_model():
    model = Sequential()
    model.add(Conv2D(16,(5,5), strides= (1,1), activation ='relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation ='relu'))
    model.add(Dense(10, activation = 'softmax'))
    
    #compile model
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
model= convolutional_model()

model.fit(X_train,Y_train, validation_data = (X_val, Y_val), epochs = 10, batch_size = 100, verbose = 2)
results = model.predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results, name= "Label")
submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), results], axis = 1)
submission.to_csv("cnn_mnist_datagen.csv", index = False)
