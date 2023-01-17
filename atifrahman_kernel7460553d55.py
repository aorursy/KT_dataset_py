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

train_data = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')
y = train_data[['label']]

train_data = train_data.drop(['label'], axis = 1)

X = train_data

X_test = test_data
#Normalizing the data

X = X / 255.0

X_test = X_test / 255.0
#Reshaping the data from 28 * 28 to 28 * 28 *1

X = X.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)
#One hot encoding

from tensorflow.keras.utils import to_categorical

y = to_categorical(y, num_classes = 10)
#DATA AUGMENTATION

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_gen = ImageDataGenerator(

               rotation_range = 40,

               width_shift_range = 0.2,

               height_shift_range = 0.2,

               shear_range = 0.2,

               zoom_range = 0.2,

               horizontal_flip = False,

               vertical_flip = False,

               fill_mode = 'nearest')



image_gen.fit(X)
#display multiple images using subplot

import numpy as np

from matplotlib import pyplot as plt

rows = 4

cols = 3

ax = []

fig = plt.figure(figsize = (10,10))

for i in range(rows * cols):

    ax.append(fig.add_subplot(rows, cols, i+1))

    ax[-1].set_title("Label" + str(y[i]))

    plt.imshow(X[i][:,:,0], cmap = 'gray')

fig.tight_layout()

plt.show()    
#split the data into training and validation sets

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.1, random_state = 4)
# CNN architechture



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.callbacks import ReduceLROnPlateau



model = keras.Sequential(

     [

         layers.Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (28,28,1)),

         layers.Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu'),

         layers.MaxPool2D(pool_size = (2,2)),

         layers.Dropout(0.25),

         

         layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'),

         layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'),

         layers.MaxPool2D(pool_size = (2,2), strides = (2,2)),

         layers.Dropout(0.25),

         

         layers.Flatten(),

         layers.Dense(256, activation = 'relu'),

         layers.Dropout(0.50),

         layers.Dense(10, activation = 'softmax')

         

     ])
#optimizers

optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#compiling the model by choosing optimizers, loss and metrics.

model.compile(

    optimizer = optimizer,

    loss = 'categorical_crossentropy',

    metrics = ['accuracy']

)
# set a learning rate annealer

learning_rate_annealer = ReduceLROnPlateau(

                             monitor = 'val_acc',

                             patience = 3,

                             verbose = 1,

                             factor = 0.5,

                             min_lr = 0.00001)

#set epochs and batch size

epochs = 30

batch_size = 64
#Model checkpoint callback

model_checkpoint = keras.callbacks.ModelCheckpoint(

                           'my_checkpoint.h5', monitor = 'val_loss', save_best_only = True)
'''history = model.fit_generator( image_gen.flow(X_train, Y_train, batch_size = batch_size),

                               epochs = epochs,

                               validation_data = (X_val, Y_val),

                               verbose = 2,

                               steps_per_epoch = int(X_train.shape[0]//batch_size),

                               callbacks = [learning_rate_annealer,

                                            model_checkpoint])'''

                                            
# saving model using joblib

'''from sklearn.externals import joblib

joblib.dump(best_model, 'kaggle_model.pkl')'''


filename = '../input/checkpoint/my_checkpoint.h5'

best_model = keras.models.load_model(filename)
#plot loss and accuracy curves

'''

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color = 'b', label = 'Training loss')

ax[0].plot(history.history['val_loss'], color = 'r', label = 'Validation loss')

legend = ax[0].legend(loc = 'best', shadow = True)



ax[1].plot(history.history['acc'], color = 'b', label = 'Training Accuracy')

ax[1].plot(history.history['val_acc'], color = 'r', label = 'Validation Accuracy')

legend = ax[1].legend(loc = 'best', shadow = True)'''
#plot confusion matrix

import seaborn as sns

from sklearn.metrics import confusion_matrix

import numpy as np

Y_pred = best_model.predict(X_val)

Y_pred_classes = np.argmax(Y_pred, axis = 1)

Y_true = np.argmax(Y_val, axis = 1)

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

sns.heatmap(confusion_mtx, annot = True, fmt = 'd')

#predict the results 

results = best_model.predict(X_test)



results = np.argmax(results, axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)