# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Extraccion de los datos de entrenamiento 

train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()


test= pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
# Se prepara la data 

Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)

X_train = X_train / 255.0

X_test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)
X_train
Y_train
import matplotlib.pyplot as plt

# vemos las imagenes

plt.figure(figsize=(15,4.5))

for i in range(15):  

    plt.subplot(3, 10, i+1)

    plt.imshow(X_train[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
# se  crea mas imagenes

datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)
# imagenes aumentadas

X_train3 = X_train[9,].reshape((1,28,28,1))

Y_train3 = Y_train[9,].reshape((1,10))

plt.figure(figsize=(15,4.5))

for i in range(10):  

    plt.subplot(3, 10, i+1)

    X_train2, Y_train2 = datagen.flow(X_train3,Y_train3).next()

    plt.imshow(X_train2[0].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')

    if i==9: X_train3 = X_train[11,].reshape((1,28,28,1))

    if i==19: X_train3 = X_train[18,].reshape((1,28,28,1))

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
# construyendo la red neuronal

nets = 10

model = [0] *nets

for j in range(nets):

    model[j] = Sequential()



    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.4))



    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.4))



    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Flatten())

    model[j].add(Dropout(0.4))

    model[j].add(Dense(10, activation='softmax'))



    # usando adam optimizer

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# entrenando las redes

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

history = [0] * nets

epochs = 45

for j in range(nets):

    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)

    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),

        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  

        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)

    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
# se hace una grafica

plt.plot(history[0].history['acc'])

plt.plot(history[0].history['val_acc'])

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train','Test'])

plt.show()
# otra grafica

plt.plot(history[0].history['loss'])

plt.plot(history[0].history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
# ensamblamos las predicciones

results = np.zeros( (X_test.shape[0],10) ) 

for j in range(nets):

    results = results + model[j].predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("MNIST-CNN-ENSEMBLE.csv",index=False)