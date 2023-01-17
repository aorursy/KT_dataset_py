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
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow  import keras
import math
# Installing required libraries
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import random
import os
#
# Seed for random weights
#

seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
data = pd.read_csv(os.path.join(dirname, filename))
data.head(10)
data.info()

plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True,  square=True, cmap='coolwarm')
plt.show()

features_mean= list(data.columns[1:11])

color_dic = {'M':'red', 'B':'blue'}
colors = data['diagnosis'].map(lambda x: color_dic.get(x))
sm = pd.plotting.scatter_matrix(data[features_mean], c=colors, alpha=0.4, figsize=((15,15)));
plt.show()
color_dic = {'M':1, 'B':0}

data['diagnosis_encoded'] = data['diagnosis'].map(lambda x: color_dic.get(x))
#
# Dynamic learning rate, tensorflow board logging, earlystop, Checkpoints
#

def scheduler(epoch):
  epoch_limit = 3
  if epoch < epoch_limit:
    return 0.01
  else:
    return  max(0.001 * math.exp(0.001 * (epoch_limit - epoch)) , 0.0001)



lrcallback = tf.keras.callbacks.LearningRateScheduler(scheduler)

import os, datetime

logdir = os.path.join("logs3", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


# {epoch:02d}-{val_loss:.2f}
checkpoint_filepath = './{epoch:02d}-{val_accuracy:.2f}.checkpoint.hdf5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                              filepath=checkpoint_filepath,
                              monitor='val_accuracy',
                              mode='max',
                              save_best_only=True)



# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)

cols = ['id', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

X = data[cols].to_numpy()

Y = data.diagnosis_encoded.to_numpy()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.001)))
model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Nadam(), metrics=["accuracy"])
model.fit(x_train, y_train , validation_data=( x_test, y_test), batch_size=1, epochs=25, verbose=1, callbacks=[ model_checkpoint_callback, tensorboard_callback, lrcallback])
