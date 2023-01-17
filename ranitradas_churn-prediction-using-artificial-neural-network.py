#Importing Libraries

import numpy as np

import pandas as pd
#Loading the Dataset

df = pd.read_csv('../input/churn-prediction/Churn_Modelling.csv')

df.head()
#Extracting the Dependent & Independent Variables

X = df.iloc[:,3:13].values

Y = df.iloc[:,13].values



print(X)

print(Y)
#To create Dummy Variables

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:,2] = le.fit_transform(X[:,2])

print(X)
#Creating separate columns for each Country as the potential Dummy Variable would have had 3 levels

#Thereby, nullifying any sort of ordinalities

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

col_tr = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])],remainder = 'passthrough')

X = np.array(col_tr.fit_transform(X))

print(X)
#Remove Dummy Variable Trap

X = X[:,1:]

print(X)

#Splitting the Data into Train & Test Set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
import tensorflow as tf

ann = tf.keras.models.Sequential()



# Input and First Hidden Layers

ann.add(tf.keras.layers.Dense(units = 6, activation ='relu'))



#Second Hidden Layer

ann.add(tf.keras.layers.Dense(units = 6, activation ='relu'))



#Output Layer

ann.add(tf.keras.layers.Dense(units = 1, activation ='sigmoid'))
#Compilation

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Training the Model

ann.fit(X_train, y_train, batch_size = 40, epochs = 100)
#Prediction

y_pred = ann.predict(X_test)

print(y_pred)
#To get the predicted classes (0 or 1) corresponding to the observed classes

y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#Creating Confusion Matrix and measuring the test accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

con_mat = confusion_matrix(y_test, y_pred)

print('Confusion Matrix:\n',con_mat)



accuracy = accuracy_score(y_test, y_pred)

print('Accuracy of the Model on the Test Set:\t',accuracy)