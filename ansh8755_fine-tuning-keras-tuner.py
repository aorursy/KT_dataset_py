# Install keras tuner
!pip install -U keras-tuner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
from tensorflow import keras

from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
# Load Fashion MNIST Dataset

mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) =mnist.load_data()
train_images = train_images/255
test_images = test_images/255
plt.imshow(train_images[0])
train_images.shape, test_images.shape
train_images = train_images.reshape(train_images.shape[0], 28,28,1)
test_images = test_images.reshape(test_images.shape[0], 28,28,1)
train_images.shape, test_images.shape
def build_model(hp):
    model = keras.Sequential([
        keras.layers.Conv2D(hp.Int('Conv1_filter', min_value=32, max_value=256, step=32), #To tune the number of filters (argument "step" is the step size)
                        hp.Choice('Conv1_filtersize', values=[3,5]),  #To tune shape of filter
                       activation='relu',
                       input_shape=(28,28,1)),
        
        keras.layers.Conv2D(hp.Int('Conv2_filter', min_value=32, max_value=512, step=32),
                        hp.Choice('Conv2_filtersize', values=[3,5]),
                       activation='relu'),
        
        keras.layers.Dropout(hp.Choice('Dropout_1', values=[0.0, 0.10,0.20, 0.30, 0.40])),  #To tune the level of dropout

        keras.layers.MaxPooling2D(2,2), 
    
        keras.layers.Flatten(),
        keras.layers.Dense(hp.Int('Dense1', min_value=128, max_value=512, step=32),
                          activation='relu'),
        keras.layers.Dropout(hp.Choice('Dropout_1', values=[0.0, 0.10,0.20, 0.30, 0.40])),
        
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),  #To tune the learning rate
                 loss='sparse_categorical_crossentropy',
                 metrics= ['accuracy'])
    
    return model
# Initialize RandomSearch
tuner_search = RandomSearch(build_model,    #callable that takes hyperparameters and returns a Model instance
                            objective='val_accuracy',   #String. Name of model metric to minimize or maximize
                            max_trials=5,   #Total number of trials
                            directory='output')   #To save checkpoints
tuner_search.search(train_images, train_labels, epochs=3, validation_split=0.15)
tuner_search.search_space_summary()
model = tuner_search.get_best_models(num_models=1)[0]  #Models are available in the form of list
model.summary()
model.fit(train_images, train_labels, epochs=10, validation_split=0.15, initial_epoch=3)
prediction = model.predict(test_images)
prediction = np.argmax(prediction, axis=1)
prediction
from sklearn.metrics import accuracy_score
accuracy_score(test_labels, prediction)