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


!ls ../input/resnet50/
!pwd
!cp ../input/resnet50/* /tmp/.keras/models/
!ls ~/.keras/models/
train = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')
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

Train_path = '../input/airbus-ship-detection/train_v2/'
Test_path = '../input/airbus-ship-detection/test_v2/'
%%time
training_img_data = []
target_data = []
from PIL import Image
data = np.empty((len(train_sample['ImageId']),256, 256,3), dtype=np.uint8)
data_target = np.empty((len(train_sample['ImageId'])), dtype=np.uint8)
image_name_list = os.listdir(Train_path)
index = 0
for image_name in image_name_list:
    if image_name in list(train_sample['ImageId']):
        imageA = Image.open(Train_path+image_name).resize((256,256)).convert('RGB')
        data[index]=imageA
        data_target[index]=train_sample[train_gp['ImageId'].str.contains(image_name)]['exist_ship'].iloc[0]
        index+=1
        
print(data.shape)
print(data_target.shape)
from sklearn.preprocessing import OneHotEncoder
targets =data_target.reshape(len(data_target),-1)
enc = OneHotEncoder()
enc.fit(targets)
targets = enc.transform(targets).toarray()
print(targets.shape)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(data,targets, test_size = 0.2)
x_train.shape, x_val.shape, y_train.shape, y_val.shape
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(
    rescale=1./255,
    zca_whitening = False,
    rotation_range = 90,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    brightness_range = [0.5, 1.5],
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True
    
)
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

from keras.applications.resnet50 import ResNet50
img_width, img_height = 256, 256
model = ResNet50(weights = 'imagenet', include_top=False, input_shape = (img_width, img_height, 3))
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model 
for layer in model.layers:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)
from keras import optimizers
epochs = 10
lrate = 0.001

model_final.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_final.summary()
fbeta_metric_callback = FBetaMetricCallback(beta=2)
history = model_final.fit_generator(img_gen.flow(x_train, y_train, batch_size = 16),steps_per_epoch = len(x_train)/16, validation_data = (x_val,y_val), epochs = epochs, callbacks=[fbeta_metric_callback] )

#gc.collect()
history.history
from matplotlib import pyplot

pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.plot(history.history['val_fbeta'])
pyplot.show()
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.show()
train_predict_sample = train_gp.sample(2000)
print(train_predict_sample['exist_ship'].value_counts())

image_test_name_list = os.listdir(Train_path)
number_sample = 20000
data_test = np.empty((number_sample,256, 256,3), dtype=np.uint8)
test_name = []
index = 0
for image_name in image_test_name_list:
    imageA = Image.open(Train_path+image_name).resize((256,256)).convert('RGB')
    test_name.append(image_name)
    data_test[index]=imageA
    index+=1
    if number_sample == index:
        break
print (data_test.shape)
len(data_test)
len(test_name)
gc.collect()
result = model_final.predict(data_test)
!rm -f Have_ship_or_not.csv
!ls -alrt
result_list={
    "ImageId": test_name,
    "Have_ship":np.argmax(result,axis=1)
}
result_pd = pd.DataFrame(result_list)
result_pd.to_csv('Have_ship_or_not.csv',index = False)
