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
import tensorflow as tf
import pandas as pd

bank = pd.read_csv('../input/predicting-churn-for-bank-customers/Churn_Modelling.csv')                

bank
bank.info()
bank[bank.isnull().any(axis=1)]
#Since IDs are unique to each customer, those can be removed for model building



bank1= bank.drop(columns=['RowNumber','CustomerId','Surname'])

bank1
# One hot encoding for the multi-class variable 'Geography'



bank1=pd.get_dummies(data=bank1, columns=['Geography'])
# Since Gender is binary, let's encode them



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

bank1['Gender'] = le.fit_transform(bank1['Gender'])





bank1
#Distinguishing Feature and Target set



# Since we're to identify whether the customer will leave or not, our target would be exited and other columns would be our features that help determine the target



X=bank1.drop(columns=['Exited'])





y=bank1['Exited'].to_numpy()
#Divide the data into train and test set



# Split into Train and Test set in the ratio of 7:3



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)
# Since the target values are of class type, we need to categorize them

y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)

y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
#Normalize the train and test set





from sklearn.preprocessing import Normalizer

transformer = Normalizer()

X_train = transformer.fit_transform(X_train)

X_test = transformer.fit_transform(X_test)
y_train
y_test
#Initialize and Build the model



from keras.layers import Dense



# Model Building

#Initialize Sequential Graph (model)

model1 = tf.keras.Sequential()



#Normalize the data

model1.add(tf.keras.layers.BatchNormalization())



#Add Dense layer for prediction - Keras declares weights and bias automatically

#model1.add(tf.keras.layers.Dense(10, activation='sigmoid'))



#Output layer with 2 neurons as we have two distinct Class values

model1.add(tf.keras.layers.Dense(2, activation='softmax'))



# Model compilation with SGD and Cross-Entropy

model1.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])
# Model Fit and Predict



model1.fit(X_train,y_train,validation_data=(X_test, y_test), epochs=100)
model1.evaluate(X_test,y_test)
#Model Optimisation



# Model Building

# Let's nuilding one more hidden layer



#Initialize Sequential Graph (model)

model2 = tf.keras.Sequential()



#Normalize the data

model2.add(tf.keras.layers.BatchNormalization())



#Add Dense layer for prediction - Keras declares weights and bias automatically

model2.add(tf.keras.layers.Dense(100, activation='sigmoid'))



#Add 2nd Dense layer with 8 neurons

model2.add(tf.keras.layers.Dense(100, activation='sigmoid'))



#Output layer with 2 neurons as we have two distinct Class values

model2.add(tf.keras.layers.Dense(2, activation='softmax'))



# Model compilation with SGD and Cross-Entropy

model2.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])
model2.fit(X_train,y_train,validation_data=(X_test, y_test), epochs=100)
# Model with 0.5 as the threshold or learning rate



# Model Building

#Initialize Sequential Graph (model)



from tensorflow.keras.optimizers import SGD

model3 = tf.keras.Sequential()



#Normalize the data

model3.add(tf.keras.layers.BatchNormalization())



#Add Dense layer for prediction - Keras declares weights and bias automatically

model3.add(tf.keras.layers.Dense(10, activation='sigmoid',input_shape=(13,)))



#Output layer with 2 neurons as we have two distinct target values

model3.add(tf.keras.layers.Dense(2, activation='softmax'))



# Model compilation with SGD and Cross-Entropy

sgd = SGD(lr=0.5)

model3.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])
model3.fit(X_train,y_train,validation_data=(X_test, y_test), epochs=50)
model3.evaluate(X_test,y_test)
#Predicting the results with 0.5 as threshold





y_pred = model3.predict(X_test)



#y_pred1=y_pred.round()

y_pred
#Confuson Matrix



# Creating the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

cm
# Accuracy of each Model built



# Model with just Input and Output

print ("Accuracy of Model with just Input and Output is",model1.evaluate(X_test,y_test));



# Model with Input, two hidden layers and Output

print ("Accuracy of Model with Input, two hidden layers and Output is",model2.evaluate(X_test,y_test));



# Model with Input, one hidden layer and Output with 0.5 threshold

print ("Accuracy of Model with Input, one hidden layer and Output with 0.5 threshold is",model3.evaluate(X_test,y_test));