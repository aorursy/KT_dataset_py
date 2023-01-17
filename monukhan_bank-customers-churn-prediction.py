import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/bank-customers/Churn Modeling.csv')

dataset.head()
dataset.isnull().sum()
dataset.info()
dataset.describe()
value_counts = pd.value_counts(dataset['Exited'])

plt.figure(figsize = (6,6))

value_counts.plot(kind = 'pie', explode = [0,0.1],autopct='%1.1f%%', shadow=True)

plt.title('Proportion of customer churned and retained')

plt.show()

value_counts
sns.countplot(dataset['Geography'])

plt.title('Geographical location Distribution of Bank Customers')

plt.show()
sns.countplot(dataset['Gender'])

plt.title('Gender Distribution of Bank Customers')

plt.show()


fig, axarr = plt.subplots(2, 2, figsize=(20, 12))

sns.countplot(x='Geography', hue = 'Exited',data = dataset, ax=axarr[0][0])

sns.countplot(x='Gender', hue = 'Exited',data = dataset, ax=axarr[0][1])

sns.countplot(x='HasCrCard', hue = 'Exited',data = dataset, ax=axarr[1][0])

sns.countplot(x='IsActiveMember', hue = 'Exited',data = dataset, ax=axarr[1][1])
sns.pairplot(dataset, hue = 'Exited')
plt.figure(figsize = (15,15))

sns.heatmap(dataset.corr(), annot = True, cmap = 'RdYlGn')
X = dataset.iloc[:,3:-1].values

y = dataset.iloc[:,-1].values
X
y
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:, 2] = le.fit_transform(X[:, 2])
print(X)
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
print(X_train)
print(y_train)


import tensorflow as tf
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
ann.compile(optimizer  ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model_history = ann.fit(X_train, y_train,validation_split=0.33,batch_size = 10, epochs = 50)
print(model_history.history.keys())
plt.plot(model_history.history['accuracy'])

plt.plot(model_history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.show()


y_pred = ann.predict(X_test)

y_pred =(y_pred > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))