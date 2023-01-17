# Import libraries

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Read the data file

df = pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
#Check the first 5 rows

df.head()
# Watch the correlation of the label or the target coumlumn

df.corr()['diagnosis'].sort_values()
# Visualize the correlation with kind heatmap

sns.heatmap(df.corr())
# Becaue the label is categorical (0,1) we can use countplot to check it there is

# balance between the count values , is this case it is some kind balanced , wich is good.

sns.countplot(x='diagnosis', data=df)
# Another way to visulize the target (label) correlation is using bar plot,

# after dorping the label itself ('diagnosis')

df.corr()['diagnosis'][:-1].sort_values().plot(kind='bar')
# Create the X,y , and usint the values of it because Tensorflow use the array instead ot dataframe

X = df.drop('diagnosis', axis=1).values

y = df['diagnosis'].values
# Model preprocessing 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)
# check the shape ot data sets .

print('X_train :',X_train.shape)

print('X_test :',X_test.shape)

print('-'*20)

print('X_train :',X_train.shape)

print('X_test :',X_test.shape)
# Import the scaler

from sklearn.preprocessing import MinMaxScaler
# Intiate an instance from the scaler

scaler = MinMaxScaler()
# Scale and fit the X_train 

X_train = scaler.fit_transform(X_train)



# Scale y_train withour fiting , to avoid data leakage. 

X_test = scaler.transform(X_test)
# check training set after scaling .

X_train
import tensorflow as ts

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dropout
#Create the model

model = Sequential()



#make the layers , begine with Dense 5 , because the shape of the traing set is 5.

model.add(Dense(5, activation='relu'))

model.add(Dense(3, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



# use loss = binary_crossentropy , because our label is binary ( 0,1 ).

model.compile(loss='binary_crossentropy', optimizer='adam')
# fit the model .

# use validation_data , to compare the prediction results lates with the test results.

model.fit(x=X_train,

          y=y_train,

          validation_data=(X_test,y_test),

          epochs = 300,verbose=1)
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report , confusion_matrix
print(classification_report(y_test, predictions))

# we can see that the accuracy is 95 %
print(confusion_matrix(y_test,predictions))

# See that we mis 3 values from 60