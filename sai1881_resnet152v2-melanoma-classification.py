# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pydicom
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

from IPython.display import HTML
import cv2


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, Flatten, Dropout, Dense, Conv1D, MaxPooling1D, BatchNormalization, GRU
import tensorflow_datasets as tfds
from tensorflow import feature_column
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelBinarizer
import matplotlib.pyplot as plt
#
# Sample image
#

image_string = tf.io.read_file("../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0015719.jpg")
image=tf.image.decode_jpeg(image_string,channels=3)

# Image shape 
(image.shape, image.numpy().max())
#
# Sample image
#
fig = plt.figure()
plt.subplot(1,2,1)
plt.title('Original image')
plt.imshow(image)
#
# Load meta
#

data = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
for ind, row in data.iterrows():
    data.loc[ind, "image_path"] = row.image_name + ".jpg"
    
data = data[['patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'diagnosis', 'benign_malignant', 'target', 'image_path']]
data.target = data.target.astype('str')


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)
# UNDER SAMPLING to remove imbalance and decrese data
data_mod = pd.concat([data[data.target == "1"][0:584], data[data.target == "0"][0:500]]).sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)
#
# image Augmentation
#

train_image_generator = ImageDataGenerator(
                                            featurewise_center=False, samplewise_center=False,
                                            featurewise_std_normalization=False, samplewise_std_normalization=False,
                                            zca_whitening=False, zca_epsilon=1e-06, rotation_range=40, width_shift_range=0.2,
                                            height_shift_range=0.0, brightness_range=None, shear_range=0.2, zoom_range=0.2,
                                            channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=True,
                                            vertical_flip=True, rescale=1. / 255, preprocessing_function=None,
                                            data_format=None, validation_split=0.2, dtype=None
                                        )
batch_size = 16
IMG_SHAPE = (1024, 1024, 3)
IMG = (IMG_SHAPE[0], IMG_SHAPE[1])
train_data_gen = train_image_generator.flow_from_dataframe(
                                                            data_mod, directory="../input/siim-isic-melanoma-classification/jpeg/train", x_col='image_path', y_col='target', weight_col=None,
                                                            target_size=IMG, color_mode='rgb', classes=None,
                                                            class_mode='binary', batch_size=batch_size, shuffle=True, seed=None,
                                                            save_to_dir=None, save_prefix='', save_format='png', subset="training",
                                                            interpolation='nearest', validate_filenames=True
                                                        )
Counter(train_data_gen.classes)
test_data_gen = train_image_generator.flow_from_dataframe(
                                                            data_mod, directory="../input/siim-isic-melanoma-classification/jpeg/train", x_col='image_path', y_col='target', weight_col=None,
                                                            target_size=IMG, color_mode='rgb', classes=None,
                                                            class_mode='binary', batch_size=batch_size, shuffle=True, seed=None,
                                                            save_to_dir=None, save_prefix='', save_format='png', subset="validation",
                                                            interpolation='nearest', validate_filenames=True
                                                        )
Counter(test_data_gen.classes)

#
# RES-NET MODEL Config
#
inception_model = tf.keras.applications.ResNet152V2(
    include_top=True, weights='imagenet', input_tensor=None,
    pooling=None, classes=1000, classifier_activation='softmax'
)

# Enable Training of resnet
for layer in inception_model.layers:
    layer.trainable = True


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc')
]

model = Sequential()
model.add(inception_model.layers[-2])
model.add(Flatten())
model.add(Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=[METRICS])
keras.backend.set_value(model.optimizer.lr,0.1)
epochs = 10
class_weight = dict(Counter(train_data_gen.classes))
total = len(train_data_gen.classes)
print(dict(Counter(train_data_gen.classes)))
class_weight = {i:(1/j)*total/len(class_weight) for i,j in class_weight.items()}
class_weight
import math

# Dynamic learning rate
def scheduler(epoch):
  epoch_limit = 5

  if epoch < epoch_limit:
    return 0.001
  else:
    return  max(0.0001 * math.exp(0.0001 * (epoch_limit - epoch)) , 0.0001)
lrcallback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(
  train_data_gen,
  steps_per_epoch=len(train_data_gen.filepaths) // batch_size,
  epochs=20,
  verbose=1,
  validation_data=test_data_gen,
  validation_steps=len(test_data_gen.filepaths) // batch_size,
  class_weight=class_weight,
  callbacks=[lrcallback])
model.save("res2t.h5")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
