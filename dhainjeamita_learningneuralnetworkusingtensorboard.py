import tensorflow as tf

%load_ext tensorboard.notebook

tensorboard_callback = tf.keras.callbacks.TensorBoard("myLogs")
from random import random

from numpy import array

from numpy import cumsum

from matplotlib import pyplot

from pandas import DataFrame

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import TimeDistributed

from keras.layers import Bidirectional

from keras.utils.vis_utils import plot_model
!rm -rf ./myLogs/ 
import pandas as pd

dataset = pd.read_csv("../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv", header=None)

X = dataset.iloc[:,0:8]

y = dataset.iloc[:,8]

# define the keras model

model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# compile the keras model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset

model.fit(X, y, epochs=1, batch_size=1,  callbacks=[tensorboard_callback])

# evaluate the keras model

_, accuracy = model.evaluate(X, y)

print('Accuracy: %.2f' % (accuracy*100))
%tensorboard --logdir myLogs
!ls -ltr /kaggle/working/myLogs
!cat /kaggle/working/myLogs/events.out.tfevents.1570724722.796002eb8092