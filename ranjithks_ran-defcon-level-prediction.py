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
df_Train = pd.read_csv('/kaggle/input/hackerearth-defcon/train.csv')

df_Test = pd.read_csv('/kaggle/input/hackerearth-defcon/test.csv')



df_Sub = pd.read_csv('/kaggle/input/hackerearth-defcon/sample_submission.csv')
df_Train.sample(5)
df_Test.sample(5)
df_Train.drop(columns=['ID'], axis=1, inplace=True)

df_Test.drop(columns=['ID'], axis=1, inplace=True)
df_Train.shape
df_Test.shape
df_Train_new = pd.get_dummies(df_Train, columns=['Allied_Nations', 'Hostile_Nations', 'Diplomatic_Meetings_Set', 'Aircraft_Carriers_Responding'])

df_Test_new = pd.get_dummies(df_Test, columns=['Allied_Nations', 'Hostile_Nations', 'Diplomatic_Meetings_Set', 'Aircraft_Carriers_Responding'])
X_Train = df_Train_new.drop(columns=['DEFCON_Level'], axis=1)

y_Train = df_Train_new['DEFCON_Level']



X_Test = df_Test_new.copy()
X_Train.shape
X_Test.shape
X_Train.sample(5)
X_Test.sample(5)
y_Train.value_counts()
from sklearn.preprocessing import MinMaxScaler
mmScalerX = MinMaxScaler()



X_Train_s = mmScalerX.fit_transform(X_Train)

X_Test_s = mmScalerX.fit_transform(X_Test)
from imblearn.over_sampling import SMOTE



X_Train_resampled, y_Train_resampled = SMOTE(random_state=25).fit_resample(X_Train_s, y_Train)
X_Train_resampled.shape
y_Train_resampled.shape
y_Train_resampled.value_counts()
import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import regularizers



print(tf.__version__)
import matplotlib.pyplot as plt
!pip install git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs

import tensorflow_docs.plots

import tensorflow_docs.modeling
def build_model():

  model = keras.Sequential([layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001), input_shape=[X_Train_resampled.shape[1]]),

                            layers.Dropout(0.2),

                            layers.Dense(384, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),

                            layers.Dropout(0.2),

                            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),

                            layers.Dropout(0.2),

                            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),

                            layers.Dropout(0.2),

                            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),

                            layers.Dropout(0.2),

                            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),

                            layers.Dropout(0.2),

                            layers.Dense(6, activation='softmax')

                            ])



  # Defining the optimizer with a specific learning rate of 0.001

  optimizer = tf.keras.optimizers.Adam(0.001, amsgrad=True)



  # Compiling the model

  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model
model = None

del model

model = build_model()
model.summary()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
EPOCHS=500

BATCHES=64
history = model.fit(X_Train_resampled, y_Train_resampled, epochs=EPOCHS, batch_size=BATCHES, validation_split=0.2,

                    verbose=2, callbacks=[early_stop, tfdocs.modeling.EpochDots()])
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric="loss")

plt.ylim([0, 2])

plt.ylabel('Loss')
plotter.plot({'Basic': history}, metric="accuracy")

plt.ylim([0, 2])

plt.ylabel('Accuracy')
model.evaluate(X_Train_resampled, y_Train_resampled, verbose=2, batch_size=BATCHES)
y_pred = np.argmax(model.predict(X_Test_s), axis=1)
X_Test_s.shape
y_pred.shape
df_Sub.shape
df_Sub