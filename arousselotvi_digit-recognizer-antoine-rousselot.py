  # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Lambda
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

#Running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



# we import the training set:
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()


# we import the test set:
test= pd.read_csv("../input/test.csv")
print(test.shape)
print(test.values)
test.head()
#We create the test and train vectors
Y_train = train["label"] # label digits
print (Y_train[0])
X_train = train.drop(labels = ["label"],axis = 1) # pixels values, we just drop the label column
print(X_train.values)

#normalization of the data set in grayscale
X_train = X_train / 255.0
test = test / 255.0


#reshaping in 28x28:
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#We transform the labels into vectors => i.e 5 becomes [0,0,0,0,0,1,0,0,0,0]
#To do that, we'll use the to_categorical function from keras that does exactly that
print(Y_train[0])
Y_train = to_categorical(Y_train, num_classes = 10)
print(Y_train[0])
#Let's set a CNN model. Here we'll start with the following architecture: [Conv2D->Maxpooling->Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))



#Now that the CNN model layers are set, we'll set up an optimizer. RMSprop seems to have pretty good performances
#with low learning rates:
rmsOptimizer = RMSprop(lr=0.001)
#and we compile our model:
model.compile(loss = "categorical_crossentropy", metrics=["accuracy"], optimizer = rmsOptimizer)
print(model.summary())
#Fit the model. We'll use only 3 epochs for obvious computational reasons 
print(X_train.shape)
history = model.fit(X_train, Y_train, batch_size = 128, epochs = 3, 
          validation_split=0.1, verbose = 2)
#Let's plot the loss and accuracy:
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,1)#Separate in two plots
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
#Eventually, let's predict our results:
results = model.predict(test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)