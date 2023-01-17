import numpy as np 

import tensorflow as tf 

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model

from keras.layers import Dense, Activation, Flatten

from keras.models import Sequential
train_datagen = ImageDataGenerator(

#         rescale=1./255,

#         zoom_range=0.2,

        horizontal_flip=False)



train_generator = train_datagen.flow_from_directory(

    directory='/kaggle/input/image-data-with-valid/Data_Loader_Dataset/Train',

    target_size=(224, 224),

    color_mode="rgb",

    batch_size=64,

    class_mode="binary",

    shuffle=True,

    seed=42

)
valid_datagen = ImageDataGenerator(

#         rescale=1./255,

#         zoom_range=0.2,

        horizontal_flip=False)





valid_generator = valid_datagen.flow_from_directory(

    directory='/kaggle/input/image-data-with-valid/Data_Loader_Dataset/Valid',

    target_size=(224, 224),

    color_mode="rgb",

    batch_size=64,

    class_mode="binary",

    shuffle=True,

    seed=42

)
model = Sequential([

    keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling='max'),

#     Flatten(),

    Dense(512),

    Activation('relu'),

    Dense(1),

    Activation('sigmoid')

])

model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
history = model.fit_generator(

        train_generator,

        validation_data=valid_generator,

        epochs=5, verbose = 1)
def eval_metric(model, history, metric_name):

    '''

    Function to evaluate a trained model on a chosen metric. 

    Training and validation metric are plotted in a

    line chart for each epoch.

    

    Parameters:

        history : model training history

        metric_name : loss or accuracy

    Output:

        line chart with epochs of x-axis and metric on

        y-axis

    '''

    metric = history.history[metric_name]

    val_metric = history.history['val_' + metric_name]

    e = range(1, 5 + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)

    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)

    plt.xlabel('Epoch number')

    plt.ylabel(metric_name)

    plt.title('Comparing training and validation ' + metric_name)

    plt.legend()

    plt.show()
import matplotlib.pyplot as plt

eval_metric(model, history, 'loss')
from keras.models import load_model

model.save('ResNet50_valid.h5')
import pickle 

filename = 'ResNet50_pickle.pkl'

model = pickle.dump(model, open(filename, 'wb'))