import numpy as np

import pandas as pd

import seaborn as sns

import tensorflow as tf



import os

import cv2

import random



import matplotlib.pyplot as plt
from keras.models import Sequential

from keras.layers import Dense, Dropout,Activation, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



from sklearn.model_selection import train_test_split
os.listdir("../input/notMNIST_small/notMNIST_small")
Datadir = '../input/notMNIST_small/notMNIST_small'

Categories = ['A', 'D', 'I', 'G', 'H', 'B', 'F', 'C', 'J', 'E']



img_size = 80

training_data = []



for Category in Categories:

    path = os.path.join(Datadir, Category)

    print("Loading data from ", path)

    class_num = Categories.index(Category)

    for img in os.listdir(path):

        try:

            img_array = cv2.imread(os.path.join(

                path, img), cv2.IMREAD_GRAYSCALE)

            img_resize = cv2.resize(img_array, (img_size, img_size))

            training_data.append([img_resize, class_num])

        except Exception as e:

            pass
random.shuffle(training_data)



X, y = [], []

for feature, label in training_data:

    X.append(feature)

    y.append(label)



X = np.array(X).reshape(-1, img_size, img_size, 1)

X = X / 255.0



y = to_categorical(y, num_classes = 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
print("full data:  ",X.shape)

print("Train:      ",X_train.shape)

print("Validation: ",X_val.shape)

print("Test:       ",X_test.shape)
conv_layers = [1,2,3] # X2

conv_layer_nodes = [16,32,64]

dense_layers = [1,2]

dense_layer_nodes = [64,128]



batch_size = 64

epochs = [10]



models = []

models_names = []

models_histories =[]

models_scores = []



# 3 Conv with 64 nodes, 1 Dense with 265 nodes, trained for 10 epochs and batch_size of 64  Score:  95.08809397008034



for conv_layer in conv_layers:

    for conv_layer_node in conv_layer_nodes:

        for dense_layer in dense_layers:

            for dense_layer_node in dense_layer_nodes:

                for epoch in epochs:

                    Name = "{} Conv with {} nodes, {} Dense with {} nodes".format(conv_layer*2,conv_layer_node,dense_layer, dense_layer_node)

                    

                    print(Name)

                    model = Sequential()



                    model.add(Dropout(0.2, input_shape=(80,80,1)))

                    

                    model.add(Conv2D(int(conv_layer_node/2.0), (5, 5)))

                    model.add(Activation("relu"))

                    model.add(BatchNormalization())



                    model.add(Conv2D(conv_layer_node, (5, 5)))

                    model.add(Activation("relu"))

                    model.add(BatchNormalization())



                    model.add(MaxPool2D(pool_size=(2, 2)))

                    model.add(Dropout(0.25)) 

                    

                    #model.add(MaxPool2D(pool_size=(2, 2)))



                    for l in range(conv_layer-1):

                        model.add(Conv2D(conv_layer_node, (3, 3)))

                        model.add(Activation("relu"))

                        model.add(BatchNormalization())



                        model.add(Conv2D(int(conv_layer_node*2.0), (3, 3)))

                        model.add(Activation("relu"))

                        model.add(BatchNormalization())



                        model.add(MaxPool2D(pool_size=(2, 2)))

                        model.add(Dropout(0.25)) 



                    model.add(Flatten())



                    for l in range(dense_layer):

                        model.add(Dense(dense_layer_node))

                        model.add(Activation("relu"))

                        model.add(BatchNormalization())



                        model.add(Dropout(0.5))  





                    model.add(Dense(10))

                    model.add(Activation("softmax"))



                    model_path = './Model.h5'





                    model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

                    # Set a learning rate annealer

                    callbacks = [

                        EarlyStopping(monitor='val_acc', patience=20, mode='max', verbose=1),

                        ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True, mode='max', verbose=1),

                        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

                    ]





                    # Without data augmentation i obtained an accuracy of 0.98114

                    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epoch, 

                                        validation_data = (X_val, y_val), verbose = 0)

                    

                    score = model.evaluate([X_test], [y_test], verbose=0)

                    print("Score: ",score[1]*100)



                    models_names.append(Name)

                    models.append(model)

                    models_histories.append(history)

                    models_scores.append(score[1])

for i in range(len(models)):

    history = models_histories[i]

    Model_Name = models_names[i]

    Model_score = models_scores[i]

    

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(1, len(acc) + 1)



    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(Model_Name)



    ax1.plot(epochs, acc, label='Training acc')

    ax1.plot(epochs, val_acc, label='Validation acc')

    ax1.legend()

    ax2.plot(epochs, loss,  label='Training loss')

    ax2.plot(epochs, val_loss, label='Validation loss')

    ax2.legend()



    print("Model: ",Model_Name," Score: ",Model_score*100)



    plt.show()
best_model = models[models_scores.index(max(models_scores))]

Best_Model_Name = models_names[models_scores.index(max(models_scores))]



print(Best_Model_Name)

history = best_model.fit(X_train, y_train, batch_size = batch_size, epochs = 100, 

                    validation_data = (X_val, y_val), verbose = 1, callbacks = callbacks)

                    

score = best_model.evaluate([X_test], [y_test], verbose=0)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle(Best_Model_Name)



ax1.plot(epochs, acc, label='Training acc')

ax1.plot(epochs, val_acc, label='Validation acc')

ax1.legend()

ax2.plot(epochs, loss,  label='Training loss')

ax2.plot(epochs, val_loss, label='Validation loss')

ax2.legend()



print("Model: ",Name," Score: ",score[1]*100)



plt.show()