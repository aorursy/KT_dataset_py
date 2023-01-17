

import keras



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.datasets import fashion_mnist



import matplotlib.pyplot as plt

import random



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")



from keras.preprocessing.image import img_to_array

import random

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv2D,BatchNormalization

from keras.layers import MaxPool2D,Dropout

from keras.layers import Flatten

from keras.layers import Dense

import cv2

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()



# 60000（28×28）images

print ("x_train", x_train.shape)

print ("x_test", x_test.shape)

import seaborn as sns

g = sns.countplot(y_train)

indices = {}

for i in range(10):

    indices[i]=[]

    for j in range(len(y_train)):



        if y_train[j] == i:

            indices[i].append(j)
p = np.random.randint(1, 6000, size=(9))

fig = plt.figure(figsize=(16, 16))



label = 0



for i in range(len(p)):

    

  # plt.imshow(x_train[p[i]])

  image = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])

  image.imshow(x_train[indices[label][p[i]]])

  image.set_title("True label :{}".format((y_train[indices[label][p[i]]])))

            


p = np.random.randint(1, 60000, size=(9))

fig = plt.figure(figsize=(16, 16))

for i in range(len(p)):

    

  # plt.imshow(x_train[p[i]])

  image = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])

  image.imshow(x_train[p[i]])
# define color channel to 1

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)



# define input data to CNN（28×28×1）

input_shape = (28, 28, 1)



# transform for Deep Learning

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255




# Conver labels to One-Hot vector

num_classes = 10

y_train_dl = keras.utils.to_categorical(y_train, num_classes)

y_test_dl = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()



model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))



model.add(Flatten())

model.add(Dense(units=128,activation="relu"))

# model.add(Dense(units=64,activation="relu"))

model.add(Dense(units=num_classes, activation="softmax"))

model.summary()







from keras import optimizers

model.compile(loss="categorical_crossentropy",

              optimizer=optimizers.Adam(),

              metrics=["accuracy"])
# https://keras.io/callbacks/#reducelronplateau

# Set a learning rate annealer

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=1, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



epochs = 10

history = model.fit(x_train, y_train_dl,

                    batch_size=64, #Number of simultaneous learning

                    epochs=epochs, #Number of train

                    verbose=1, # halfway output

                    validation_data=(x_test, y_test_dl),

                   callbacks=[learning_rate_reduction]) # data for validation

epochs = 10

history = model.fit(x_train, y_train_dl,

                    batch_size=64, #Number of simultaneous learning

                    epochs=epochs, #Number of train

                    verbose=1, # halfway output

                    validation_data=(x_test, y_test_dl)) # data for validation
## Visuarize train history



def plot_history(history):

    # print(history.history.keys())

    from matplotlib import pyplot as plt



    # accuracy

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.legend(['acc', 'val_acc'], loc='lower right')

    plt.show()



    # loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.legend(['loss', 'val_loss'], loc='lower right')

    plt.show()

    

    ## saveimage

    plt.savefig("history.png")



# plot

plot_history(history)

!ls

#download

#from google.colab import files

#files.download("history.png")


y_pred = model.predict(x_test)

y_pred = [np.argmax(x) for x in y_pred]



## Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5,6,7,8,9])

print (cm)



from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred))
def check(list1, val): 

      

    # traverse in the list 

    for x in list1:

        if val == x: 

            return True

    return False
labelss=['0','1','2','3','4','5','6','7', '8', '9']



major_F = [0, 2, 4, 6]



wrong_index = []



for i in range(len(y_pred)):

    if (y_pred[i]) != (y_test[i]):

        if check(major_F, y_test[i]):

            wrong_index.append(i)





im_idx = random.sample(wrong_index, k=9)



nrows = 3

ncols = 3

fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))



n = 0

for row in range(nrows):

    for col in range(ncols):

            ax[row,col].imshow(x_test[im_idx[n]].reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(labelss[(y_pred[im_idx[n]])], labelss[(y_test[im_idx[n]])]))

            n += 1



plt.show()
def show_label(label):

    p = np.random.randint(1, 6000, size=(9))

    fig = plt.figure(figsize=(16, 16))



    # label = 0



    for i in range(len(p)):



      # plt.imshow(x_train[p[i]])

      image = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])

      image.imshow(x_train[indices[label][p[i]]].reshape(28,28))

      image.set_title("True label :{}".format((y_train[indices[label][p[i]]])))
show_label(0)
show_label(1)
show_label(2)
show_label(3)
show_label(4)
show_label(5)
show_label(6)
show_label(7)
show_label(8)
show_label(9)