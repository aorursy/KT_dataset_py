# Based on Tarun Kumar's notebook 
# Imports



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import cv2 as cv



from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D

from keras import models

from keras.optimizers import Adam, RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



import pickle



%matplotlib inline
# Loading and Visualising Dataset



## Seed to resample the same permutation every time

np.random.seed(1)

df_train = pd.read_csv("../input/digit-recognizer/train.csv")



## Load dataset

df_train = df_train.iloc[np.random.permutation(len(df_train))]

df_train.head(5)
df_train.shape
# Preparing training and validation data



## Set sizes

sample_size = df_train.shape[0]

validation_size = int(df_train.shape[0]*0.1)





# train_x and train_y

train_x = np.asarray(df_train.iloc[:sample_size-validation_size,1:]).reshape([sample_size-validation_size,28,28,1]) # taking all columns expect column 0

train_y = np.asarray(df_train.iloc[:sample_size-validation_size,0]).reshape([sample_size-validation_size,1]) # taking column 0



# val_x and val_y

val_x = np.asarray(df_train.iloc[sample_size-validation_size:,1:]).reshape([validation_size,28,28,1])

val_y = np.asarray(df_train.iloc[sample_size-validation_size:,0]).reshape([validation_size,1])

train_x.shape, train_y.shape
# Loading test.csv



df_test = pd.read_csv("../input/digit-recognizer/test.csv")

test_x = np.asarray(df_test.iloc[:,:]).reshape([-1,28,28,1])

# Normalize pixel data



train_x = train_x/255

val_x = val_x/255

test_x = test_x/255

model = models.Sequential()
# Keras Sequential modelling



## Block 1

model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))

model.add(LeakyReLU())

model.add(Conv2D(32,3, padding  ="same"))

model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



## Block 2

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(256,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(10,activation="sigmoid"))
# Compiling the model





initial_lr = 0.001

loss = "sparse_categorical_crossentropy"

model.compile(Adam(lr=initial_lr), loss=loss ,metrics=['accuracy'])

model.summary()
# Training



epochs = 7

batch_size = 256

history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_data=(val_x,val_y))

# Training performance



## Diffining Figure

f = plt.figure(figsize=(20,7))



## Adding Subplot 1 (For Accuracy)

f.add_subplot(121)



plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



## Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()
test_y = np.argmax(model.predict(test_x),axis =1)
# Evaluate the model on the test data using `evaluate`

print("Evaluate on test data")

results = model.evaluate(test_x, test_y, batch_size=128)

print("test loss, test acc:", results)



# Generate predictions (probabilities -- the output of the last layer)

# on new data using `predict`

print("Generate predictions for 3 samples")

predictions = model.predict(test_x[:3])

print("predictions shape:", predictions.shape)