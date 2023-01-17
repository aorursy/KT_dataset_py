import pandas as pd
import numpy as np

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard

import time
#read csv file
df = pd.read_csv("../input/train.csv")

#data pre-processing
category_train = df["label"]
train = df.drop(["label"], axis = 1)

#normalize the data
train = train/255

#reshape the data
train = train.values.reshape(-1, 28, 28, 1)

#one hot encode categorical data
category_train = to_categorical(category_train, num_classes = 10)
#initialize parameters for model optimization
conv_layers = [1, 2, 3]
dense_layers = [0, 1, 2]
layer_sizes = [32, 64]

#Initialize callbacks to minimize loss
learning_rate_reduction = ReduceLROnPlateau(factor = 0.5, patience = 1, monitor = "val_loss")
'''
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(name)
            
            tensorboard = TensorBoard(log_dir = 'logs/{}'.format(name))
            
            model = Sequential()
            
            model.add(Conv2D(layer_size, kernel_size = (6,6), padding = "same", activation = "relu", input_shape = train.shape[1:]))
            model.add(MaxPool2D(pool_size = (2,2)))
            
            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, kernel_size = (6,6), padding = "same", activation = "relu"))
                model.add(MaxPool2D(pool_size = (2,2)))
                
            model.add(Flatten())
            
            for l in range(dense_layer):
                model.add(Dense(layer_size, activation = "relu"))
                model.add(Dropout(0.4))
                
            model.add(Dense(10, activation = "softmax"))
            
            model.compile(optimizer = Adam(), loss = "categorical_crossentropy", metrics = ["accuracy"])
            
            model.fit(train, category_train, epochs = 10, verbose = 0, batch_size = 32, validation_split = 0.2, callbacks = [tensorboard])
            '''