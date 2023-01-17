import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Input
from keras.optimizers import Adam,RMSprop
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/wine-quality/winequalityN.csv')
data.head()
data.columns.values
cols_to_norm = ['fixed acidity', 'volatile acidity', 'citric acid','residual sugar', 'chlorides', 'free sulfur dioxide','total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
data['type'].unique()
for i in range(len(data)):
    if data.iloc[i]['type'] == 'white':
        data.at[i,'type'] = 0.0
    if data.iloc[i]['type'] == 'red':
        data.at[i,'type'] = 1.0
data.isnull().sum()
data = data.dropna()
data.isnull().sum()
data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min())/(x.max() - x.min()))
data.head()
X = data[cols_to_norm]
X = np.asarray(X)
X[:5]
y = data['type']
y = np.asarray(y)
from keras.utils import to_categorical
y = to_categorical(y,2)
y[:5]
X[0].shape
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True)
print(len(X_train),len(X_test))
#Input data shape: (11,)
input_layer = Input(shape=(11,))
dense_layer = Dense(24,activation='relu')(input_layer)
dropout = Dropout(0.5)(dense_layer)
dense_layer = Dense(2,activation='sigmoid')(dropout)
model = keras.Model(input_layer,dense_layer,name='classifier')
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
model.fit(X_train,y_train,epochs=15,validation_data=(X_test,y_test))
predictions = model.predict(X_test)
predictions[:5]
predictions = np.argmax(predictions,axis=1)
validation = np.argmax(y_test,axis=1)
conf = confusion_matrix(validation,predictions)
conf_matrix = pd.DataFrame(conf, index = ['White','Red'],columns = ['White','Red'])
#Normalizing
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize = (18,15))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})