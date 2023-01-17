import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.model_selection import train_test_split
from time import time
#from tensorflow.keras.layers import  SimpleRNN 
from tensorflow.keras import regularizers
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow as tf;print("tensorflow:",tf.__version__)
from tensorflow.keras import layers
#directory = '/kaggle/input/Polymerase512vec/'
df1= pd.read_csv('/kaggle/input/polymerase512vec/Train1941_512vectorsWithTitles.csv')
df1.head()
df1.shape
df1.dtypes
df1.describe()
# check missing values
df1.isnull().sum()

df1.hist(column='Group', bins=50)
df_imputed=df1.drop(['title', 'abstract'], axis=1)
df_imputed

# we want to predict the X and y data as follows:

if 'Group' in df_imputed:
    y = df_imputed['Group'].values # get the labels we want
    del df_imputed['Group'] # get rid of the class label
    X = df_imputed.values # use everything else to predict
type(y)
type(X)
# features=
# #it's good practice to Scale Data
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_train = scaler.fit_transform(X)
# scaled_train_df = pd.DataFrame(scaled_train, columns=features)
# cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1776)
type(X_test)

print(len(y_test))
print(len(y_train))
from tensorflow.keras.callbacks import TensorBoard
tb = TensorBoard(log_dir=f"logs\\{time()}")

tf.compat.v1.disable_eager_execution() #disable eager execution,since got error AttributeError: Tensor.graph is meaningless when eager execution is enabled.
FEATURES=512
weight_decay=1e-5
model = tf.keras.Sequential([
    layers.Dense(300, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                 #kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dropout(0.1),
        layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    #layers.Dense(units, activation=activation_func),
    #layers.SimpleRNN(100,unroll=True),
    layers.Dense(6 ,activation='softmax')
])

callback = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_crossentropy', patience=10)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=74, batch_size=100)
model2 = tf.keras.Sequential([
    layers.Dense(300, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                 #kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dropout(0.1),
        layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    #layers.Dense(units, activation=activation_func),
    #layers.SimpleRNN(100,unroll=True),
    layers.Dense(6 ,activation='softmax')
])

callback = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_crossentropy', patience=10)

model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=1000)
model3 = tf.keras.Sequential([
    layers.Dense(300, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                 #kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dropout(0.1),
        layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(300, activation='relu'),
    layers.Dropout(0.1),
    #layers.Dense(units, activation=activation_func),
    #layers.SimpleRNN(100,unroll=True),
    layers.Dense(6 ,activation='softmax')
])

callback = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_crossentropy', patience=10)

model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model3.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=1000)
X_test.shape
 
y_pred_keras = model2.predict(X_test)[:, :]
y_pred_keras
max_index = np.argmax(y_pred_keras, axis=1)
max_index
y_test
max_index-y_test
sum((max_index-y_test)==0)/X_test.shape[0] #verify the method to calculate accuracy, confirm the number matches val loss output
df2= pd.read_csv('/kaggle/input/polymerase512vecholdout/HoldOut388_512vectors2WithTitles.csv')
df2.head()
df_imputed2=df2.drop(['title', 'abstract'], axis=1)
df_imputed2
x_test2=df_imputed2
x_test2.shape
y_pred_keras2 = model2.predict(x_test2)[:, :]
y_pred_keras2
max_indexMLP = np.argmax(y_pred_keras2, axis=1)
max_indexMLP
## compare MLP with RF predictions

# RF results:
# preds[0:20] 
# Out[48]:
# array([[0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 1, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 1, 0],
#        [0, 0, 0, 0, 1, 0],
#        [0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1]], dtype=uint8)
# [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 1, 1, 4, 4, 1, 5, 5],

# MLP results 1-20
# [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 1, 1, 4, 4, 1, 5, 5, 
# RF preds[21:40]

# Out[51]:
# array([[0, 1, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1],
#        [0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 1, 0],
#        [0, 0, 0, 0, 0, 1],
#        [0, 1, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1],
#        [0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 1, 0],
#        [0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1],
#        [0, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 1, 0]], dtype=uint8)
#([1, 1, 5, 1, 5, 5, 4, 5, 1, 1, 1, 5, 1, 4, 1, 5, 1, 5, 4],

# MLP results 21-40
# 5, 1, 1, 5, 1, 5, 5, 4, 5, 1, 1, 1, 5, 1, 4, 1, 5, 1, 5, 4,
max_indexRF=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 1, 1, 4, 4, 1, 5, 5, 5, 1,
       1, 5, 1, 5, 5, 4, 5, 1, 1, 1, 5, 1, 4, 1, 5, 1, 5, 4, 3, 1, 5, 5,
       1, 4, 1, 0, 1, 4, 1, 1, 0, 4, 0, 1, 5, 0, 4, 1, 5, 1, 5, 3, 5, 5,
       1, 5, 5, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 1, 4, 5, 1, 1, 5, 1,
       1, 1, 1, 1, 1, 1, 0, 1, 5, 1, 1, 1, 1, 5, 0, 0, 5, 4, 1, 0, 1, 1,
       4, 1, 0, 4, 4, 1, 4, 4, 3, 0, 1, 0, 0, 4, 1, 0, 5, 1, 3, 1, 3, 1,
       0, 5, 1, 5, 3, 5, 1, 1, 1, 5, 4, 0, 1, 1, 4, 4, 4, 0, 4, 1, 0, 0,
       1, 0, 0, 3, 5, 2, 0, 1, 1, 0, 4, 1, 3, 0, 0, 5, 1, 1, 0, 0, 5, 1,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 4, 5, 0, 5, 1, 0, 1,
       5, 0, 5, 1, 5, 1, 1, 5, 1, 5, 1, 0, 5, 0, 0, 0, 1, 0, 0, 1, 1, 0,
       5, 1, 1, 1, 5, 0, 5, 1, 1, 0, 1, 0, 1, 5, 1, 4, 5, 1, 0, 1, 1, 5,
       1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 5, 0, 5, 0, 1, 5, 5, 5, 0, 0, 1,
       4, 1, 5, 5, 0, 0, 5, 1, 5, 5, 1, 0, 5, 4, 0, 5, 1, 5, 1, 1, 1, 0,
       1, 1, 5, 5, 1, 5, 5, 1, 1, 0, 1, 0, 0, 5, 0, 5, 1, 1, 1, 5, 0, 0,
       5, 0, 0, 0, 5, 1, 5, 0, 5, 5, 1, 0, 5, 5, 5, 1, 5, 1, 1, 0, 0, 5,
       1, 5, 5, 1, 5, 0, 1, 1, 5, 5, 1, 5, 5, 1, 5, 5, 5, 5, 0, 0, 5, 1,
       1, 0, 0, 1, 5, 0, 0, 1, 1, 5, 1, 5, 5, 1, 5, 5, 0, 0, 0, 5, 1, 1,
       0, 0, 0, 4, 5, 0, 5, 1, 0, 1, 0, 5, 0, 0]
max_indexRF
max_indexMLP-max_indexRF
sum((max_indexMLP-max_indexRF)==0)/x_test2.shape[0]
max_indexXGB=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 1, 1, 4, 4, 1, 5, 5, 5, 1,
       1, 5, 5, 5, 5, 4, 5, 1, 1, 1, 5, 1, 4, 1, 5, 1, 5, 4, 3, 4, 5, 5,
       1, 4, 1, 2, 1, 4, 1, 1, 0, 4, 0, 1, 5, 0, 4, 1, 5, 1, 5, 1, 5, 5,
       1, 5, 5, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 1, 4, 5, 1, 1, 5, 5,
       5, 1, 1, 1, 1, 1, 1, 1, 5, 1, 5, 1, 1, 5, 0, 0, 5, 4, 1, 0, 1, 1,
       4, 1, 0, 4, 4, 1, 4, 4, 1, 0, 1, 4, 3, 4, 1, 0, 1, 1, 5, 1, 3, 1,
       1, 5, 1, 5, 3, 5, 1, 1, 1, 5, 4, 0, 1, 1, 4, 4, 4, 0, 4, 1, 0, 0,
       1, 0, 0, 1, 5, 1, 0, 1, 1, 0, 4, 1, 3, 0, 0, 5, 0, 1, 0, 0, 5, 1,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 4, 5, 0, 5, 1, 0, 1,
       5, 0, 5, 1, 5, 1, 1, 1, 2, 5, 5, 0, 5, 1, 3, 0, 1, 4, 0, 1, 1, 5,
       5, 1, 1, 1, 5, 0, 5, 1, 1, 3, 1, 0, 1, 5, 1, 4, 5, 1, 5, 1, 1, 5,
       1, 5, 0, 0, 1, 2, 5, 0, 0, 5, 1, 5, 2, 5, 1, 1, 5, 5, 5, 1, 0, 1,
       4, 1, 5, 5, 5, 5, 5, 1, 5, 1, 1, 1, 5, 4, 0, 5, 1, 5, 1, 2, 2, 0,
       1, 1, 5, 5, 1, 5, 5, 5, 2, 4, 1, 5, 1, 5, 0, 5, 1, 2, 1, 5, 5, 2,
       5, 5, 1, 2, 5, 4, 5, 0, 5, 5, 1, 0, 5, 5, 5, 1, 5, 5, 1, 3, 3, 5,
       5, 5, 5, 1, 5, 5, 1, 1, 5, 5, 1, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 1,
       1, 5, 5, 1, 5, 5, 0, 1, 1, 3, 1, 5, 5, 1, 5, 5, 2, 5, 2, 5, 1, 1,
       5, 5, 3, 4, 5, 5, 5, 1, 3, 1, 5, 5, 0, 0]
sum((max_indexMLP-max_indexXGB)==0)/x_test2.shape[0]
sum((max_indexRF-max_indexXGB)==0)/x_test2.shape[0]
type(max_indexMLP)
type(max_indexRF)
type(max_indexXGB)
npRF= np.asarray(max_indexRF, dtype=np.float32)
type(npRF)
sum((npRF-max_indexXGB)==0)/x_test2.shape[0]
npRF-max_indexXGB

dfAll = np.array([[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 1, 1, 4, 4, 1, 5, 5, 5, 1,
       1, 5, 1, 5, 5, 4, 5, 1, 1, 1, 5, 1, 4, 1, 5, 1, 5, 4, 3, 1, 5, 5,
       4, 4, 1, 1, 1, 4, 1, 1, 0, 4, 0, 1, 5, 0, 4, 1, 5, 1, 5, 3, 5, 5,
       1, 5, 5, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 4, 4, 1, 4, 5, 1, 1, 5, 1,
       1, 1, 1, 1, 1, 1, 0, 1, 5, 1, 1, 1, 1, 5, 0, 0, 5, 4, 1, 0, 1, 1,
       4, 1, 0, 4, 4, 1, 4, 4, 3, 0, 1, 3, 3, 4, 1, 0, 5, 1, 3, 0, 3, 1,
       0, 5, 1, 5, 3, 1, 1, 1, 1, 5, 4, 0, 1, 1, 4, 4, 4, 0, 4, 1, 0, 0,
       1, 0, 0, 3, 5, 2, 0, 1, 1, 0, 4, 1, 1, 0, 0, 5, 1, 1, 0, 0, 5, 1,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 4, 5, 0, 5, 1, 0, 1,
       5, 0, 1, 1, 5, 1, 1, 5, 2, 5, 5, 0, 5, 1, 3, 0, 1, 4, 0, 1, 1, 5,
       5, 0, 1, 1, 5, 0, 5, 1, 2, 3, 1, 0, 1, 5, 1, 4, 5, 1, 5, 1, 2, 5,
       1, 5, 0, 0, 1, 2, 5, 0, 0, 5, 1, 5, 2, 5, 5, 1, 5, 5, 5, 3, 0, 1,
       4, 1, 5, 5, 1, 5, 5, 1, 5, 1, 1, 1, 5, 4, 0, 5, 1, 5, 1, 2, 2, 0,
       1, 1, 5, 5, 1, 5, 5, 3, 2, 4, 1, 1, 5, 5, 0, 5, 1, 2, 1, 5, 1, 2,
       5, 5, 1, 2, 5, 4, 5, 0, 5, 5, 1, 0, 5, 5, 5, 1, 5, 1, 1, 3, 3, 3,
       5, 5, 5, 1, 5, 5, 1, 1, 5, 5, 1, 5, 5, 1, 1, 5, 5, 5, 5, 5, 5, 1,
       5, 5, 5, 1, 5, 5, 0, 1, 1, 3, 1, 1, 5, 1, 5, 5, 2, 5, 2, 5, 1, 1,
       5, 5, 3, 3, 5, 5, 5, 1, 1, 1, 5, 5, 3, 0], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 1, 1, 4, 4, 1, 5, 5, 5, 1,
       1, 5, 1, 5, 5, 4, 5, 1, 1, 1, 5, 1, 4, 1, 5, 1, 5, 4, 3, 1, 5, 5,
       1, 4, 1, 0, 1, 4, 1, 1, 0, 4, 0, 1, 5, 0, 4, 1, 5, 1, 5, 3, 5, 5,
       1, 5, 5, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 1, 4, 5, 1, 1, 5, 1,
       1, 1, 1, 1, 1, 1, 0, 1, 5, 1, 1, 1, 1, 5, 0, 0, 5, 4, 1, 0, 1, 1,
       4, 1, 0, 4, 4, 1, 4, 4, 3, 0, 1, 0, 0, 4, 1, 0, 5, 1, 3, 1, 3, 1,
       0, 5, 1, 5, 3, 5, 1, 1, 1, 5, 4, 0, 1, 1, 4, 4, 4, 0, 4, 1, 0, 0,
       1, 0, 0, 3, 5, 2, 0, 1, 1, 0, 4, 1, 3, 0, 0, 5, 1, 1, 0, 0, 5, 1,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 4, 5, 0, 5, 1, 0, 1,
       5, 0, 5, 1, 5, 1, 1, 5, 1, 5, 1, 0, 5, 0, 0, 0, 1, 0, 0, 1, 1, 0,
       5, 1, 1, 1, 5, 0, 5, 1, 1, 0, 1, 0, 1, 5, 1, 4, 5, 1, 0, 1, 1, 5,
       1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 5, 0, 5, 0, 1, 5, 5, 5, 0, 0, 1,
       4, 1, 5, 5, 0, 0, 5, 1, 5, 5, 1, 0, 5, 4, 0, 5, 1, 5, 1, 1, 1, 0,
       1, 1, 5, 5, 1, 5, 5, 1, 1, 0, 1, 0, 0, 5, 0, 5, 1, 1, 1, 5, 0, 0,
       5, 0, 0, 0, 5, 1, 5, 0, 5, 5, 1, 0, 5, 5, 5, 1, 5, 1, 1, 0, 0, 5,
       1, 5, 5, 1, 5, 0, 1, 1, 5, 5, 1, 5, 5, 1, 5, 5, 5, 5, 0, 0, 5, 1,
       1, 0, 0, 1, 5, 0, 0, 1, 1, 5, 1, 5, 5, 1, 5, 5, 0, 0, 0, 5, 1, 1,
       0, 0, 0, 4, 5, 0, 5, 1, 0, 1, 0, 5, 0, 0],[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 1, 1, 4, 4, 1, 5, 5, 5, 1,
       1, 5, 5, 5, 5, 4, 5, 1, 1, 1, 5, 1, 4, 1, 5, 1, 5, 4, 3, 4, 5, 5,
       1, 4, 1, 2, 1, 4, 1, 1, 0, 4, 0, 1, 5, 0, 4, 1, 5, 1, 5, 1, 5, 5,
       1, 5, 5, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 1, 4, 5, 1, 1, 5, 5,
       5, 1, 1, 1, 1, 1, 1, 1, 5, 1, 5, 1, 1, 5, 0, 0, 5, 4, 1, 0, 1, 1,
       4, 1, 0, 4, 4, 1, 4, 4, 1, 0, 1, 4, 3, 4, 1, 0, 1, 1, 5, 1, 3, 1,
       1, 5, 1, 5, 3, 5, 1, 1, 1, 5, 4, 0, 1, 1, 4, 4, 4, 0, 4, 1, 0, 0,
       1, 0, 0, 1, 5, 1, 0, 1, 1, 0, 4, 1, 3, 0, 0, 5, 0, 1, 0, 0, 5, 1,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 4, 5, 0, 5, 1, 0, 1,
       5, 0, 5, 1, 5, 1, 1, 1, 2, 5, 5, 0, 5, 1, 3, 0, 1, 4, 0, 1, 1, 5,
       5, 1, 1, 1, 5, 0, 5, 1, 1, 3, 1, 0, 1, 5, 1, 4, 5, 1, 5, 1, 1, 5,
       1, 5, 0, 0, 1, 2, 5, 0, 0, 5, 1, 5, 2, 5, 1, 1, 5, 5, 5, 1, 0, 1,
       4, 1, 5, 5, 5, 5, 5, 1, 5, 1, 1, 1, 5, 4, 0, 5, 1, 5, 1, 2, 2, 0,
       1, 1, 5, 5, 1, 5, 5, 5, 2, 4, 1, 5, 1, 5, 0, 5, 1, 2, 1, 5, 5, 2,
       5, 5, 1, 2, 5, 4, 5, 0, 5, 5, 1, 0, 5, 5, 5, 1, 5, 5, 1, 3, 3, 5,
       5, 5, 5, 1, 5, 5, 1, 1, 5, 5, 1, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 1,
       1, 5, 5, 1, 5, 5, 0, 1, 1, 3, 1, 5, 5, 1, 5, 5, 2, 5, 2, 5, 1, 1,
       5, 5, 3, 4, 5, 5, 5, 1, 3, 1, 5, 5, 0, 0]])

dfAll 
dfAll = pd.DataFrame(dfAll)

dfAll
dfAll.mode(axis=0)
dfAll.mode(axis=0).isnull().sum()
x_test2.shape[0]-dfAll.mode(axis=0).isnull().sum().sum()/2
