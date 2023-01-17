import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.__version__
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
df.head()
df.info()
X = df.iloc[:, 3: -1].values
y = df.iloc[:, -1].values
print(X.shape)
y
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:,2])
X
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
X_test.shape
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
ann = tf.keras.models.Sequential()
type(ann)
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs= 100)
ann.summary()
X_test
y_pred = ann.predict(X_test)
y_pred.shape
y_pred = y_pred>0.5
y_pred[:15]
# concatenate predictions with test
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cm
accuracy_score(y_test, y_pred)
X[0]
[1,0,0,600,1,40,3,60000,2,1,1,50000]
sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])
ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))
ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))>0.5