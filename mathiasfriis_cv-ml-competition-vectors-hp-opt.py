# --- INSTALL TALOS ---
!pip install talos
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from tensorflow import keras
from PIL import Image
import re
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras import backend as K 
from keras.layers import *
from keras.optimizers import Adam, Nadam, RMSprop
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
import talos as ta
from talos.model import *

CSV_length = 4096;
N_classes = 29;
N_train_samples = 1000
N_validation_samples = 100
N_epochs = 50


# --- DEFINITION OF PARAMETER SPACE ---
p = {'lr': (0.0001,0.01,5),
     'amsgrad' : [True, False] ,
     'first_neuron':[32, 64, 128],
     'hidden_layers':[1,2],
     'batch_size': [16, 32, 64],
     'epochs': [50],
     'dropout': (0.1, 0.5, 5),
     'shapes':['brick'],
     'activation':['relu'],
     'last_activation': ['softmax']}
# --- DEFINITION OF MODEL ARCHITECTURE ---
# first we have to make sure to input data and params into the function
def vector_model(x_train, y_train, x_val, y_val, params):

    print("Current hyperparameters:")
    print(params)
    # Input layer
    model = Sequential()
    model.add(Dense(10, input_dim=CSV_length,
                    activation=params['activation'],
                    kernel_initializer='normal'))
    
    model.add(Dropout(params['dropout']))
    
    # Number of hidden layers depend on hyperparameter
    hidden_layers(model, params, 1)
   
    # Output layer
    model.add(Dense(N_classes, activation='softmax',
                    kernel_initializer='normal'))
    
   # opt = Adam(learning_rate=params['lr'])
    opt = Adam(lr=params['lr'], amsgrad=params['amsgrad'])
    
    model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])
    
    history = model.fit(x_train, y_train, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=1)
    
    # finally we have to make sure that history object and model are returned
    return history, model
trainingLabels = pd.read_csv('../input/au-eng-cvml2020/Train/trainLbls.csv', header=None);
trainingVectors = pd.read_csv('../input/au-eng-cvml2020/Train/trainVectors.csv', header=None).T;


validationLabels = pd.read_csv('../input/au-eng-cvml2020/Validation/valLbls.csv', header=None);
validationVectors = pd.read_csv('../input/au-eng-cvml2020/Validation/valVectors.csv', header=None).T;

testVectors = pd.read_csv('../input/au-eng-cvml2020/Test/testVectors.csv', header=None).T;

vectorsPreprocessed = False;


if vectorsPreprocessed == False:
  trainingVectors = trainingVectors.to_numpy()
  trainingVectors = trainingVectors/np.amax(trainingVectors)
  trainingLabels = trainingLabels.to_numpy()-1

  validationVectors = validationVectors.to_numpy()
  validationVectors = validationVectors/np.amax(validationVectors)
  validationLabels = validationLabels.to_numpy()-1

  testVectors = testVectors.to_numpy()
  testVectors = testVectors/np.amax(testVectors)
  
  vectorsPreprocessed = True;
print(trainingVectors.shape)
# Do grid search, to find optimized hyperparameters.
from talos import Scan

t = ta.Scan(x=trainingVectors,
            y=to_categorical(trainingLabels),
            x_val = validationVectors,
            y_val = to_categorical(validationLabels),
            model=vector_model,
            fraction_limit = 0.1,
            params=p,
            experiment_name='Experiment 1'
           )
# Save result of grid search
from talos import *
Deploy(t,'exp_1',metric='accuracy')
r = Reporting(t)
r.high('val_accuracy')
#r.plot_hist('val_accuracy')
r.table('val_accuracy')
#r.plot_corr('val_accuracy',['loss', 'accuracy', 'epochs', 'round_epochs', 'val_loss'])