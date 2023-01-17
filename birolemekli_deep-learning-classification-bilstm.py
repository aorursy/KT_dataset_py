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

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import  Embedding, LSTM,Bidirectional,SimpleRNN,Conv1D
from keras import metrics, regularizers
from keras.optimizers import SGD,RMSprop, Adam, Adadelta, Adagrad ,Adamax, Nadam
import seaborn as sns
from keras.utils import np_utils
df=pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')
df=shuffle(df)
df.info()
df.describe()
df.hist(figsize=(20, 10))
df["class"].unique() 
df.isna().all()
df[df.duplicated()].count()
le = LabelEncoder()
tags = le.fit_transform(df['class'])
df['class']=tags
tags
data=df[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis']]
data['pelvic_tilt'] = data['pelvic_tilt'].apply(lambda x: data['pelvic_tilt'].mean() if x<0 else x)
data['degree_spondylolisthesis'] = data['degree_spondylolisthesis'].apply(lambda x: data['degree_spondylolisthesis'].mean() if x<0 else x)
data.head()
X_train,X_test,y_train,y_test=train_test_split(data,tags,test_size=0.15, random_state=42)
y_train1 = np_utils.to_categorical(y_train)
y_test1 = np_utils.to_categorical(y_test)
y_train1
labels = 'X_train', 'X_test'
sizes = [len(X_train), len(X_test)]
explode = (0, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')
plt.title("% Oran")
plt.show()
sns.set_palette("Reds")
correlation=df.corr()
sns.heatmap(correlation)
plt.show()
activation_func='selu'
batch_size_=16
epochs_=60
validation_split_=0.1
regularizers_lr2=0.001
verbose_=1
size=500
input_shape=6
kernel_initializer_='random_uniform'
model=Sequential()
model.add(Embedding(size,input_shape, trainable=True,input_length=input_shape))
model.add(Bidirectional(LSTM(128,activation=activation_func,kernel_initializer=kernel_initializer_,kernel_regularizer=regularizers.l2(regularizers_lr2),return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64,activation=activation_func,kernel_initializer=kernel_initializer_,kernel_regularizer=regularizers.l2(regularizers_lr2),return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64,activation=activation_func,kernel_initializer=kernel_initializer_,kernel_regularizer=regularizers.l2(regularizers_lr2),return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32,activation=activation_func,kernel_initializer=kernel_initializer_,kernel_regularizer=regularizers.l2(regularizers_lr2),return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(16,activation=activation_func,kernel_initializer=kernel_initializer_,kernel_regularizer=regularizers.l2(regularizers_lr2))))
model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6),metrics=['acc'])
#model.summary() categorical_crossentropy
history=model.fit(X_train, y_train1, epochs=epochs_, batch_size=batch_size_,verbose=verbose_,validation_split=validation_split_)
model.evaluate(X_test, y_test1, verbose=0)
fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(history.history['acc'], label='Acc')
plt.plot(history.history['val_acc'], label='Val Acc')
plt.ylabel('Acc')
plt.xlabel('Epoch Sayısı')
plt.legend(loc="upper left")
plt.show()
