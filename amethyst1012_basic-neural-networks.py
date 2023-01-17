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
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train_orig = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
train.corr()
train.isnull().any()
from pandas_profiling import ProfileReport
prof = ProfileReport(train)
prof.to_file(output_file='output.html')
train[train["Age"].isnull()]
import seaborn as sns

sns.distplot(train["Age"])
train["Age"].max()
train["Age"].min()
train["Age"].median()
train["Age"] = train["Age"].fillna(train["Age"].median())
sns.distplot(train["Age"])
train_orig["Age"].mean()
train_orig["Age"] = train_orig["Age"].fillna(train["Age"].mean())
sns.distplot(train_orig["Age"])
train[train["Embarked"].isnull()]
train["Embarked"][train["Embarked"].isnull()] ="S"
train.isnull().any()
train=train.drop("Cabin", axis=1)
train=train.drop("Name", axis=1)
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test_orig = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()
test.isnull().any()
sns.distplot(test["Age"])
test["Age"] = test["Age"].fillna(test["Age"].median())
sns.distplot(test["Age"])
test[test["Fare"].isnull()]
test["Fare"][test["Pclass"]==3].median()
test["Fare"][test["Fare"].isnull()]=test["Fare"][test["Pclass"]==3].median()
test= test.drop("Cabin",axis=1)
test= test.drop("Name",axis=1)
train.columns
test.columns
train["Sex"].unique()
test["Sex"].unique()
train.dtypes
columns_to_onehot = ["Sex", "Embarked"]
columns_to_label=["Ticket"]
train["Ticket"]
train=pd.get_dummies(train, columns=columns_to_onehot)
train
test=pd.get_dummies(test, columns=columns_to_onehot)
test
test.dtypes
from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()

train["Ticket"]=le.fit_transform(train["Ticket"])
test["Ticket"]=le.fit_transform(test["Ticket"])
pi = test["PassengerId"]
train = train.drop("PassengerId", axis=1)
test = test.drop("PassengerId", axis=1)
X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test = test
X_train
y_train
X_test
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as L



model = Sequential(name='titanic_model')

model.add(L.InputLayer(input_shape=(11,))) # necessary to use model.summary()

model.add(L.Dense(2048, activation= tf.nn.relu))
model.add(L.Dense(2048,activation = tf.nn.relu))
model.add(L.Dense(1))
# output layer, use sigmoid for binary

model.summary()

def grad(model, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model,X_train,y_train, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)
optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)
model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),optimizer = optimizer, metrics =['accuracy'])
history = model.fit(X_train,y_train,batch_size=32,epochs = 500,verbose=1,validation_split=0.2)
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'acc'], loc='upper left')
plt.show()

preds = model.predict(test)
print(preds)

submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
submission['Survived'] = [0 if pred < 0.5 else 1 for pred in preds]
submission.head(20)
from IPython.display import FileLink

submission.to_csv('submission.csv',index=False)
FileLink(r'submission.csv')
