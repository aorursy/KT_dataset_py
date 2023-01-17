import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten , Dense

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
from google.colab import files
uploaded = files.upload()
import io
df = pd.read_csv(io.BytesIO(uploaded['advertising.csv']))
df.head()
df.isnull().any()
import seaborn as sns
sns.heatmap(df.isnull())
#outliers
from sklearn.model_selection import train_test_split
x = df.drop(labels = ['Ad Topic Line','City','Timestamp'],axis = 1)
y = ['Clicked on Ad']


x = df.drop('Country',axis = 1)
len(df['Ad Topic Line'].unique())
x.head(5)
#feature standardization , however sucky the features are ewwwwww shity
#the scales are varying much and neural network cant work easily in this so we have to preprocess the feaures

x = df[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = df['Clicked on Ad']
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify = y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train
y_train .to_numpy
model = Sequential()
model.add(Dense(x.shape[1],activation='relu',input_dim = x.shape[1]))
model.add(Dense(128,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=10,epochs=20,verbose=1,validation_split=0.2)
y_pred = model.predict_classes(x_test)

model.evaluate(x_test,y_test.to_numpy())
from sklearn.ensemble import GradientBoostingClassifier
boost = GradientBoostingClassifier()
boost.fit(x_train,y_train.to_numpy())
print(boost.score(x_test,y_test.to_numpy()))
history.history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','val'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','val'])
!pip install mlxtend
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat=mat)

plot_confusion_matrix(conf_mat=mat,show_normed=True)
df.head()
df['Timestamp']
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Timestamp']
df['year'] = df['Timestamp'].dt.year
df['month'] = df['Timestamp'].dt.month
df['day'] = df['Timestamp'].dt.day
df['week'] = df['Timestamp'].dt.week
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['hour'] = df['Timestamp'].dt.hour
df['minute'] = df['Timestamp'].dt.minute
df.head(2)

#feature important
feature = df[['year','month','day','week','day_of_week','hour','minute']]
target = df['Clicked on Ad']
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(feature,target)
print(model.feature_importances_)
# some eda
import seaborn as sns
sns.jointplot(x = df['minute'], y =df['Clicked on Ad'])
sns.jointplot(x = df['day_of_week'],y = df['Clicked on Ad'],kind = 'hex')
df.columns
x_new = df[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage',   'Male', 
         'month', 'day', 'week',
       'day_of_week', 'hour', 'minute']]
y_new = df['Clicked on Ad']
x_train_new,x_test_new,y_train_new,y_test_new = train_test_split(x_new,y_new,test_size = 0.2,stratify = y_new)
x_train_new = scaler.fit_transform(x_train_new)
x_test_new = scaler.fit_transform(x_test_new)
from sklearn.ensemble import GradientBoostingClassifier
boost = GradientBoostingClassifier()
boost.fit(x_train_new,y_train_new)
print(boost.score(x_test_new,y_test_new))
model = Sequential()
model.add(Dense(x.shape[1],activation='relu',input_dim = x.shape[1]))
model.add(Dense(128,activation='relu',kernel_initializer='glorot_uniform'))
model.add(Dense(256,activation='tanh',kernel_initializer='glorot_uniform'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=10,epochs=30,verbose=1,validation_split=0.2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','val'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','val'])
#handling string data
