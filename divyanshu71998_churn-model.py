import pandas as pd 

import numpy as np 

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score
data=pd.read_csv("../input/churnmodelling/Churn_Modelling.csv")
data.head()
data.shape
x=data.iloc[:,3:-1]

y=data.iloc[:,-1]

print("x:")

print(x.head())

print("  y:")

print(y.head())
print(x.iloc[:,1].unique())

print(x.iloc[:,2].unique())
x=data.iloc[:,3:-1].values

y=data.iloc[:,-1].values

x
Le=LabelEncoder()

x[:,2]=Le.fit_transform(x[:,2])

print(Le.classes_)

ct=ColumnTransformer(transformers=[( 'OneHotEncoder',OneHotEncoder(), [1])], remainder='passthrough')

x=np.array(ct.fit_transform(x))

x
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2, random_state = 0)
sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

ann.compile(optimizer='adam',loss='binary_crossentropy' ,metrics=['accuracy'])
ann.fit(x_train,y_train,batch_size=64,epochs=200)
y_pred=ann.predict(x_test)

y_pred=(y_pred>0.5)

y_pred

cm=confusion_matrix(y_test,y_pred)

print('confusion_matrix:')

print(cm)

accuracy=accuracy_score(y_test,y_pred)

print("acuuracy:",accuracy)
new_pred=ann.predict(sc.transform(np.array([[1,0,0,600,1,40,3,60000,2,1,1,50000]])))

new_pred=(new_pred>0.5)

print(new_pred)