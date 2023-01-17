# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy,binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,classification_report
#from tensorflow.keras.intializers import
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle

%matplotlib inline
def plot_metric(history, metrics):
    metric_title = ''
    metric_y = ''
    for i in metrics:
        train_metrics = history.history[i]
        val_metrics = history.history['val_'+i]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'bo--')
        plt.plot(epochs, val_metrics, 'ro-')
        metric_title = metric_title+", "+i
        metric_y = metric_y+", "+i
    
    plt.title('Training and validation '+ metric_title)
    plt.xlabel("Epochs")
    plt.ylabel(metric_y)
    plt.legend(["train_"+metric_y, 'val_'+metric_y])
    plt.show()
#load training data
train_dataset = h5py.File('/kaggle/input/cat-not-cat/train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels
#load test data
test_dataset = h5py.File('/kaggle/input/cat-not-cat/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset['test_set_x'][:])
test_set_y_orig = np.array(test_dataset['test_set_y'][:])
#Pickle file name on to which either the model is written to or read from
c_n_c_mdl = "cat_not_cat_mdl"
#clear the backend processes and set the seed for predictable behaviour over multiple executioins / iterations
tf.random.set_seed(42)
tf.keras.backend.clear_session()
#in the multiple iterations, the data size was found to be insufficient. The model was not learning enough with the data size
#Since the development of the model was also undertaken along with the data augumentation 
#the effects of data size increase had profoud effect on the models accuracy and validation accuracy. 
#Therefore data augementation is carried out. However, it cannot be concluded that the augementation alone helped in building better model

datagen = ImageDataGenerator(  #rescale = 1.0/255,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True,
                                            rotation_range =5                                            
                                            )
#Configure the data augumentor for the dataset
iter_images= datagen.flow(train_set_x_orig,
                           train_set_y_orig,
                           batch_size = train_set_x_orig.shape[0] )
#initialise the training data 
data_set_x,data_set_y = iter_images.next()
#add data to the dataset by augumenting the exisiting dataset. multiplication of data by a factor of 10 was found to be optimal.
#too much of multiplication was found to result in overfitting of the model. Too less was resulting in lack of learning
for i in range(15):
    dx,dy = iter_images.next()
    data_set_x = np.append(data_set_x,dx,axis=0)
    data_set_y = np.append(data_set_y,dy,axis=0)
#Check for the dataset size
print(data_set_x.shape,data_set_y.shape)
# # In case older saved model is required to be loaded, uncomment this section
# model = tf.keras.load_model(c_n_c_mdl)
#Create the model 
model = Sequential()
#Add three convolution layers to extract feature map with 01 maxpool layer each in between
#the numbers have been arrived at by conducting multiple trials and observing the results.
#Varied filters size since they are learned during training
#set 01
model.add(Conv2D(filters = 256,kernel_size = 3,activation = 'relu',
                 input_shape = data_set_x.shape[1:],padding = 'valid' ))
model.add(Dropout(rate = 0.2))
model.add(Conv2D(filters = 256,kernel_size = 3,activation = 'relu',
                 padding = 'valid'))
model.add(Dropout(rate = 0.2))
model.add(MaxPooling2D(strides = (2,2),pool_size = (2,2)))
#set 02
model.add(Conv2D(filters = 128,kernel_size =3,activation = 'relu',
                 padding = 'valid'))
model.add(Dropout(rate = 0.2))
model.add(Conv2D(filters = 128,kernel_size =3,activation = 'relu',
                 padding = 'valid'))
model.add(Dropout(rate = 0.2))
model.add(MaxPooling2D(strides = (2,2),pool_size = (2,2)))
#set 03
model.add(Conv2D(filters = 64,kernel_size =3,activation = 'relu',
                 padding = 'valid'))
model.add(Dropout(rate = 0.2))
model.add(Conv2D(filters = 64,kernel_size =3,activation = 'relu',
                 padding = 'valid'))
model.add(Dropout(rate = 0.2))
model.add(MaxPooling2D(strides = (2,2),pool_size = (2,2)))
#Flatten the layer output to get a 2D data from 3D input
model.add(Flatten())
#Add NN for learning presence of an object(cat in this case) from the feature map created by convolution layer
model.add(Dense(units = 32,activation = 'selu',input_shape = data_set_x.shape[1:]))
model.add(Dropout(rate = 0.2))
#model.add(Flatten())

model.add(Dense(units = 32,activation = 'selu'))
model.add(Dropout(rate = 0.2))

# model.add(Dense(units = 32,activation = 'selu'))
# model.add(Dropout(rate = 0.5))
model.add(Dense(units = 1,activation = 'sigmoid')) #activation is sigmoid since the classification is binary classification
#Compile the model
model.compile(
        loss = 'binary_crossentropy',#sparse_categorical_crossentropy,#
        optimizer = keras.optimizers.Nadam(lr=0.0002,beta_1 = 0.9, beta_2 = 0.999),
        metrics = ['accuracy']
)

model.summary()
#Fit the data. Learn from the dataset
filepath = "best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')

history = model.fit(
            data_set_x,data_set_y,#Y training
            #batch_size =7524,
            #steps_per_epoch =len(train_set_x_orig)/32,
            #batch_size = dat_input_shape[0]  ,#Batch Size
            validation_split = 0.3  ,#validation ratio
            epochs = 30,    #epochs
            #verbose = 1,
            callbacks = [checkpoint]
)
# Generate generalization metrics
score = model.evaluate(test_set_x_orig, test_set_y_orig, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
plot_metric(history,['accuracy','loss'])
model.save(c_n_c_mdl)
train_metrics = history.history['accuracy']
epochs = range(1, len(train_metrics) + 1)
plt.plot(epochs, train_metrics, 'bo--')
train_metrics = history.history['loss']
plt.plot(epochs, train_metrics, 'ro--')

metric_title = 'Accuracy Vs Loss '
metric_y = "Accuracy & Loss"
    
plt.title('Training '+ metric_title)
plt.xlabel("Epochs")
plt.ylabel(metric_y)
plt.legend(['Accuracy','Loss'])#["train_"+metric_y, 'val_'+metric_y])
plt.show()