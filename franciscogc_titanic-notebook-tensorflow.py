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
import seaborn as sns
import matplotlib.pyplot as plt

pd_train = pd.read_csv('/kaggle/input/titanic/train.csv')
pd_train.head()

pd_test = pd.read_csv('/kaggle/input/titanic/test.csv')
pd_test.head()
pd_test.shape
pd_train['Gender'] = (pd_train['Sex'] == 'male').astype(int)
pd_train['Embarked_C'] = (pd_train['Embarked'] == 'C').astype(int)
pd_train['Embarked_Q'] = (pd_train['Embarked'] == 'Q').astype(int)
pd_train['Embarked_S'] = (pd_train['Embarked'] == 'S').astype(int)
pd_train['Age'].fillna(pd_train['Age'].mean(), inplace=True)
pd_train.head()

pd_test['Gender'] = (pd_test['Sex'] == 'male').astype(int)
pd_test['Embarked_C'] = (pd_test['Embarked'] == 'C').astype(int)
pd_test['Embarked_Q'] = (pd_test['Embarked'] == 'Q').astype(int)
pd_test['Embarked_S'] = (pd_test['Embarked'] == 'S').astype(int)
pd_test['Age'].fillna(pd_train['Age'].mean(), inplace=True)
pd_test['Fare'].fillna(pd_train['Fare'].mean(), inplace=True)
pd_test.head()
sns.pairplot(pd_train[['Age','Fare','Survived']])
sns.kdeplot(pd_train['Fare'])
plt.scatter(pd_train['Age'],pd_train['Fare'])
#pd_train['Age'].value_counts().sort_index().plot().bar(x='Age',y='Counts')
sns.kdeplot(pd_train[pd_train['Survived']==1]['Age'],cumulative=False)
sns.kdeplot(pd_train[pd_train['Survived']==0]['Age'],cumulative=False)
sns.kdeplot(pd_train[pd_train['Survived']==1]['Fare'],cumulative=False)
sns.kdeplot(pd_train[pd_train['Survived']==0]['Fare'],cumulative=False)
features = ['Gender','Age','Fare','Pclass','Embarked_C','Embarked_Q','Embarked_S','SibSp','Parch']
pd_x_train = pd_train[features]
pd_x_train.head()

pd_x_test = pd_test[features]
pd_x_test.head()
pd_x_test.isnull().sum()
# Miscellaneous operations

# pd_x_train.isnull().sum()
# type(pd_x_train[pd_x_train['Age']==22])
# np.array(pd_x_train.values)
# x_train.loc(0)[0].values()
x_train = pd_x_train.values
print("x_train =", x_train)

x_test = pd_x_test.values
print("x_test =", x_test)

y_train = pd_train['Survived'].values
print("y_train =", y_train)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
print("x_train.shape = ", x_train.shape)
print("x_val.shape = ", x_val.shape)
import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
#model.add(tf.keras.layers.Dense(200, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

history = model.fit(x_train, 
                    y_train, 
                    epochs=250,
                    batch_size=10,
                    validation_data=(x_val, y_val))
x_test.shape
pd_test['Survived'] = (np.round(model.predict(x_test))[:,0]).astype(int)

#np.abs((np.round(model.predict(x_test))[:,0] - y_test)).sum() /891
pd_test
pd_test_result = pd_test[['PassengerId', 'Survived']]
pd_test_result
pd_test_result[['PassengerId', 'Survived']].to_csv('Submission.csv', index = False)
# kaggle competitions submit -c titanic -f 'kaggle_submission.csv' -m "Titatic Notebook - TensorFlow"