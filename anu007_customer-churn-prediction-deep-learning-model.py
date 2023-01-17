import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Kaggle competition/Banking data/Churn_Modelling.csv")
data.head()
# Lets remove data Rownumber,Customer ID columns and Surname as dont provide any quantifiable Info

data = data.drop(["RowNumber","CustomerId","Surname"],axis=1)
data.head()
sns.scatterplot(data=data,x='Balance',y='CreditScore')
data.describe()
#Lets check if there are any missing values in data
data.isnull().sum()
#Lets check the data type for each variable in data

data.dtypes
#Lets check if data is biased?

data['Exited'].value_counts()
#Lets split the data into Dependant and indepandant variables arrays

target = np.array([data['Exited'].values])

print(target)
data = data.drop(['Exited'],axis=1)
#Lets isolate columns which are categorical in nature

categorical_var = data.select_dtypes(include=['object'])

categorical_var
#Lets perform Onehot encoding

from sklearn.preprocessing import OneHotEncoder
ohe_Cat_var = OneHotEncoder(sparse=False)
ohe_categorical = ohe_Cat_var.fit_transform(categorical_var)
ohe_categorical
data = data.drop(['Geography','Gender'],axis=1)
data
data = np.array(data.values)
data.ndim
data
features = data
features
features = np.concatenate((features,ohe_categorical),axis=1)
#Standardization

from sklearn.preprocessing import StandardScaler
scfeatures = StandardScaler()
features = scfeatures.fit_transform(features)
features
target = target.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=10)
#Lets check the shapes of train and test data

print('Training Set:',X_train.shape)
print('Testing Set:',X_test.shape)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)
# Check whether the model is genralized or not

print('Training Score:', classifier.score(X_train,y_train))
print('Validation Score:',classifier.score(X_test,y_test))
#let see what we could achieve with Deep learning (Try to generalize the model with layer optimizers)

#lets first convert the target array into 2d

target.ndim
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=10)
#Import Tensorflow
import tensorflow as tf

#Modelling

#Step1 : Architecting the model

model = tf.keras.models.Sequential()
#Step2: Create Input Layer

model.add(tf.keras.layers.Dense(units=70,activation='relu',input_shape=(13,)))
#Step3: Create Intermediated layers

model.add(tf.keras.layers.Dense(units=200,activation="relu"))
model.add(tf.keras.layers.Dense(units=200,activation="relu"))
model.add(tf.keras.layers.Dense(units=200,activation="relu"))
model.add(tf.keras.layers.Dense(units=200,activation="relu"))
model.add(tf.keras.layers.Dense(units=200,activation="relu"))
model.add(tf.keras.layers.Dense(units=200,activation="relu"))
model.add(tf.keras.layers.Dense(units=200,activation="relu"))
model.add(tf.keras.layers.Dense(units=200,activation="relu"))

#Step 4: Create an Output layer

model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#Step 5: Model Compilation

model.compile(optimizer='Nadam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=500, validation_data=(X_test,y_test))
