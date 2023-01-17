#!pip install pandas

#!pip install -U scikit-learn



#!python -m pip install -U pip

#!python -m pip install -U matplotlib



#!pip install pandas



#!pip install seaborn



#!pip install tensorflow
import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn



from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers



import glob

import cv2

import os











import warnings

warnings.filterwarnings('ignore')



print('Pandas:  ' + pd.__version__)

print('Numpy:   ' + np.__version__)

print('Sklearn: ' + sklearn.__version__)

print(tf.__version__)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_test = pd.read_csv('/kaggle/input/ml-for-exploration-geophysics-dnn/test.csv')

df_train = pd.read_csv('/kaggle/input/ml-for-exploration-geophysics-dnn/train.csv')
#df.head(10) #View first 10 data rows

#df.info()

df_train.describe()
df_test.describe()
feature_names = df_train.columns[:-1].tolist()

print(feature_names)

label_names = df_train.columns[-1:].tolist()

print(label_names)
sns.set()

sns.set_style("white")

sns.pairplot(df_train[1:1000], diag_kind="kde")
X_train_orig = df_train[["X", "Y", "Z"]]

X_test_orig  = df_test[["X", "Y", "Z"]]

Y_train_orig = df_train["Time"]



X_train_orig = np.asarray(X_train_orig)

Y_train_orig = np.asarray(Y_train_orig)

X_test_orig = np.asarray(X_test_orig)



print('Size of the X_train dataset: ' + str(X_train_orig.shape))

print('Size of the y_train dataset: ' + str(Y_train_orig.shape))

print('Size of the X_test dataset: ' + str(X_test_orig.shape))
scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)

scaler.fit(X_train_orig)

print("Maximum values of X_train (X, Y, Z): " + str(scaler.data_max_))

print("Minimum values of X_train (X, Y, Z): " + str(scaler.data_min_))



# Use the same transform for both train and test data:

X_train_norm = scaler.transform(X_train_orig)

X_test_norm = scaler.transform(X_test_orig)
X_train, X_val, y_train, y_val = train_test_split(X_train_norm, Y_train_orig, test_size=0.2)

print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
model = keras.Sequential([

    keras.layers.Dense(64, input_dim=3, activation='relu'),

    keras.layers.Dense(32, activation="relu"),

    keras.layers.Dense(1, activation="linear")

])



print(model.summary())
model.compile(optimizer='adam',

              loss='MSE',

              metrics=['MAE'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=1024, epochs=50)
test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)

print('\nTest loss:', test_loss)

print('\nTest accuracy:', test_acc)
print('\nhistory dict:', history.history.keys())
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)

plt.plot(history.history['MAE'])

plt.plot(history.history['val_MAE'])

plt.title('Model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.yscale('log')

plt.legend(['train', 'test'], loc='upper left')

plt.grid(True)

plt.subplot(2, 1, 2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.yscale('log')

plt.legend(['train', 'test'], loc='upper left')

plt.grid(True)

plt.show()
predictions = model.predict(X_test_norm)
sample_submission = pd.read_csv('/kaggle/input/ml-for-exploration-geophysics-dnn/sample_submission.csv')



new_submission = sample_submission

new_submission["Predicted"] = predictions



filename = 'new_submission_example.csv'

new_submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
#kaggle competitions submit -c ml-for-exploration-geophysics-dnn -f new_submission_example.csv -m "Message"