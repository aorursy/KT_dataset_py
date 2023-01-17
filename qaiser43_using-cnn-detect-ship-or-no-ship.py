# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
print(os.listdir("../input"))
import numpy as np 
import pandas as pd
import time
gc.collect()
# Any results you write to the current directory are saved as output.


train = pd.read_csv('../input/train_ship_segmentations_v2.csv')
train.head()
list(train.columns.values)
train['exist_ship'] = train['EncodedPixels'].fillna(0)
train.head()
train['exist_ship'] != 0
train.loc[train['exist_ship'] != 0 , 'exist_ship'] = 1
train.head()
del train['EncodedPixels']
train.head()
print(len(train['ImageId']))
print(train['ImageId'].value_counts().shape[0])
train_gp = train.groupby(['ImageId']).sum().reset_index()
train_gp.loc[train_gp['exist_ship']>0,'exist_ship']=1

train_sample = train_gp.sample(5000)
test_sample = train_gp.sample(1000)
print(train_gp['exist_ship'].value_counts())
print(train_sample['exist_ship'].value_counts())
print(test_sample['exist_ship'].value_counts())
print (train_sample.shape)
print (test_sample)
from keras.utils import np_utils
import numpy as np
from glob import glob

Train_path = '../input/train_v2/'
Test_path = '../input/test_v2/'
# define function to load train, test, and validation datasets
def load_dataset(path):
    files_array = []
    if str(path) == str(Train_path):
        data = np.array(train_sample['ImageId'])
        data_targets = np_utils.to_categorical(np.array(train_sample['exist_ship']), 133)

        for idx, element in  enumerate(data): 
            files_array.append(Train_path + element)

        data = np.array(files_array)
    else:
        data = np.array(test_sample['ImageId'])
        data_targets = np_utils.to_categorical(np.array(test_sample['exist_ship']), 133)

        for idx, element in  enumerate(data): 
            files_array.append(Train_path + element)

        data = np.array(files_array)
    
    return data, data_targets


# load train, test, and validation datasets
train_files, train_targets = load_dataset(Train_path)
test_files, test_targets = load_dataset(Test_path)

from keras.preprocessing import image 
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 


test_tensors = paths_to_tensor(test_files).astype('float32')/255
train_tensors = paths_to_tensor(train_files).astype('float32')/255
#https://www.kaggle.com/yassinealouini/f2-score-per-epoch-in-keras

import numpy as np 
import pandas as pd 
from keras.callbacks import Callback
from sklearn.metrics import fbeta_score
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.test_utils import get_test_data



""" F2 metric implementation for Keras models. Inspired from this Medium
article: https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
Before we start, you might ask: this is a classic metric, isn't it already 
implemented in Keras? 
The answer is: it used to be. It has been removed since. Why?
Well, since metrics are computed per batch, this metric was confusing 
(should be computed globally over all the samples rather than over a mini-batch).
For more details, check this: https://github.com/keras-team/keras/issues/5794.
In this short code example, the F2 metric will only be called at the end of 
each epoch making it more useful (and correct).
"""

# Notice that since this competition has an unbalanced positive class
# (fewer ), a beta of 2 is used (thus the F2 score). This favors recall
# (i.e. capacity of the network to find positive classes). 

# Some default constants

START = 0.5
END = 0.95
STEP = 0.05
N_STEPS = int((END - START) / STEP) + 2
DEFAULT_THRESHOLDS = np.linspace(START, END, N_STEPS)
DEFAULT_BETA = 1
DEFAULT_LOGS = {}
FBETA_METRIC_NAME = "val_fbeta"

# Some unit test constants
input_dim = 2
num_hidden = 4
num_classes = 2
batch_size = 5
train_samples = 20
test_samples = 20
SEED = 42
TEST_BETA = 2
EPOCHS = 5




# Notice that this callback only works with Keras 2.0.0


class FBetaMetricCallback(Callback):

    def __init__(self, beta=DEFAULT_BETA, thresholds=DEFAULT_THRESHOLDS):
        self.beta = beta
        self.thresholds = thresholds
        # Will be initialized when the training starts
        self.val_fbeta = None

    def on_train_begin(self, logs=DEFAULT_LOGS):
        """ This is where the validation Fbeta
        validation scores will be saved during training: one value per
        epoch.
        """
        self.val_fbeta = []

    def _score_per_threshold(self, predictions, targets, threshold):
        """ Compute the Fbeta score per threshold.
        """
        # Notice that here I am using the sklearn fbeta_score function.
        # You can read more about it here:
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
        thresholded_predictions = (predictions > threshold).astype(int)
        return fbeta_score(targets, thresholded_predictions, beta=self.beta, average='micro')

    def on_epoch_end(self, epoch, logs=DEFAULT_LOGS):
        val_predictions = self.model.predict(self.validation_data[0])
        val_targets = self.validation_data[1]
        _val_fbeta = np.mean([self._score_per_threshold(val_predictions,
                                                        val_targets, threshold)
                              for threshold in self.thresholds])
        self.val_fbeta.append(_val_fbeta)
        print("Current F{} metric is: {}".format(str(self.beta), str(_val_fbeta)))
        return

    def on_train_end(self, logs=DEFAULT_LOGS):
        """ Assign the validation Fbeta computed metric to the History object.
        """
        self.model.history.history[FBETA_METRIC_NAME] = self.val_fbeta

"""
Here is how to use this metric: 
Create a model and add the FBetaMetricCallback callback (with beta set to 2).
f2_metric_callback = FBetaMetricCallback(beta=2)
callbacks = [f2_metric_callback]
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                    nb_epoch=10, batch_size=64, callbacks=callbacks)
print(history.history.val_fbeta)
"""

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(133, activation='softmax'))

### TODO: Define your architecture.


model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint  

epochs = 20

fbeta_metric_callback = FBetaMetricCallback(beta=2)
history = model.fit(train_tensors, train_targets, 
          validation_data=(test_tensors, test_targets),
          epochs=epochs, batch_size=20, callbacks=[fbeta_metric_callback])

print(history.history['val_fbeta'])
from matplotlib import pyplot

pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.plot(history.history['val_fbeta'])
pyplot.show()
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.show()
# get index of predicted ship for each image in test set
predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
# get index of predicted ship for each image in test set
predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in train_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(train_targets, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)