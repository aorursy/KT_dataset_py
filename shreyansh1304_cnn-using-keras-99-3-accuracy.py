

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Activation,Conv2D,MaxPool2D

from keras.optimizers import SGD,Adam

from keras import regularizers

from keras.utils import np_utils

import matplotlib.pyplot as plt

train_set = pd.read_csv("../input/train.csv")

test_set = pd.read_csv("../input/test.csv")

train_set.head()
y_train = train_set['label']    # Extracting the prediction column (that the model has to predict)
x_train = train_set.drop(['label'],axis=1)

x_train = np.array(x_train)        # Converting the pandas dataframe to numpy array 

y_train = np.array(y_train)

test_set = np.array(test_set)
val_x = x_train[33000:]   #  Validation set(val_x,val_y) is used to check performance on unseen data and make 

val_y = y_train[33000:]                         # improvements
x_train = x_train[:33000]

y_train = y_train[:33000]

x_train = x_train.reshape(-1,28,28)

x_train.shape
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])                 # Printing images need data in form 28x28 and not as (33000,784)

    plt.yticks([])

    plt.grid(False)

    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()
x_train = x_train.reshape(-1,784)

x_train.shape
val_x = val_x.reshape(-1,784)

val_x.shape
test_set = test_set.reshape(28000,784)

test_set.shape
x_train = x_train/255

val_x = val_x/255

test_set = test_set/255
x_train = x_train.reshape(-1,28,28,1)

x_train.shape
val_x = val_x.reshape(-1,28,28,1)

val_x.shape
test_set = test_set.reshape(-1,28,28,1)

test_set.shape
print(np.unique(y_train, return_counts=True)) # No. of unique values in y_train(or y_val for that matter )
# one-hot encoding using keras' numpy-related utilities

n_classes = 10

print("Shape before one-hot encoding: ", y_train.shape)

y_train = np_utils.to_categorical(y_train, n_classes)

print("Shape after one-hot encoding: ", y_train.shape)

print("Shape before one-hot encoding: ", val_y.shape)

val_y = np_utils.to_categorical(val_y, n_classes)

print("Shape after one-hot encoding: ", val_y.shape)
class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 

               'Five', 'Six', 'Seven', 'Eight', 'Nine']
# Set the CNN model 





model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, 

              loss='categorical_crossentropy',

              metrics=['accuracy'])



#Note: when using the categorical_crossentropy loss, your targets should be in categorical format (e.g. if you 

# have 10 classes, the target for each sample should be a 10-dimensional vector that is all-zeros except for a 

#1 at the index corresponding to the class of the sample). In order to convert integer targets into categorical 

#targets, you can use the Keras utility to_categorical.



#When using the sparse_categorical_crossentropy loss, your targets should be integer targets. If you have categorical

# targets, you should use categorical_crossentropy.
model.fit(x_train, y_train, epochs=10, batch_size=25)
predictions = model.predict(test_set)
model.metrics_names # This list corresponds to what the list returned by model.evaluate refers to
model.evaluate(val_x,val_y)

# Accuracy on unseen data (validation set extracted from train_set)
predictions = model.predict_classes(test_set, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("CNN-MNIST.csv", index=False, header=True)