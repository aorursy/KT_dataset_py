#Import Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
#Load the dataset using pandas
dataset = pd.read_csv('/kaggle/input/titanic/train.csv')
#Data Cleaning
#Dropping irrelevant features.
df_cleaned = dataset.drop(['Survived','PassengerId','Name','Ticket','Cabin'],axis=1)
X = df_cleaned.iloc[:, :].values
y = dataset.iloc[:, 1].values
#Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, 6].reshape(-1,1))
X[:, 6] = imputer.transform(X[:, 6].reshape(-1,1)).reshape(-1,)
#Encoding for Categorical features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,4] = le.fit_transform(X[:,4])
#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Creating Neural Network with 2 hidden layers.
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fit the ANN model and predict
ann.fit(X_train, y_train, batch_size = 10, epochs = 200)
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
#Calculate metrics with the predicted results for Test set
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
