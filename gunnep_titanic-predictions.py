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
import csv
#28 x 28 images, in total 42000 for training
csv_file = '/kaggle/input/titanic/train.csv'
df = pd.read_csv(csv_file)
df.head(10)
y_train = df.Survived

df.columns 

x_train = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'],axis=1)
#clean data
x_train = x_train.fillna(-1)
#change categories to numbers
x_train.Sex = pd.Categorical(x_train.Sex)
x_train.Cabin = pd.Categorical(x_train.Cabin)
x_train.Embarked = pd.Categorical(x_train.Embarked)

x_train['Sex_cat'] = x_train.Sex.cat.codes
x_train['Cabin_cat'] = x_train.Cabin.cat.codes
x_train['Embarked_cat'] = x_train.Embarked.cat.codes

x_train.head(10)
x_train = x_train.drop(['Sex', 'Cabin','Embarked','Cabin_cat'],axis=1)

arr_y_train = y_train.to_numpy()
arr_x_train = x_train.to_numpy()
#set a boosted decision tree
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
clf.fit(arr_x_train, arr_y_train)
clf.score(arr_x_train, arr_y_train)
#set a simple NN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import roc_curve, auc, roc_auc_score

def normalize(vector):
    for f in range(vector.shape[1]):
        mean = np.mean(vector[:,f])
        std = np.std(vector[:,f])
        vector[:,f] = (vector[:,f] - mean)/std

    return vector

arr_x_train_norm = normalize(arr_x_train)
        
model = Sequential()
model.add(Dense(12, input_dim=arr_x_train.shape[1], activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(1, activation='tanh'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Fit the model
history = model.fit(arr_x_train_norm, arr_y_train, epochs=100, batch_size=256)
csv_file = '/kaggle/input/titanic/test.csv'
df = pd.read_csv(csv_file)
df.head(10)

x_test = df.drop(['PassengerId', 'Name', 'Ticket'],axis=1)
x_test = x_test.fillna(-1)
#change categories to numbers
x_test.Sex = pd.Categorical(x_test.Sex)
x_test.Cabin = pd.Categorical(x_test.Cabin)
x_test.Embarked = pd.Categorical(x_test.Embarked)

x_test['Sex_cat'] = x_test.Sex.cat.codes
x_test['Cabin_cat'] = x_test.Cabin.cat.codes
x_test['Embarked_cat'] = x_test.Embarked.cat.codes
x_test = x_test.drop(['Sex', 'Cabin','Embarked','Cabin_cat'],axis=1)

arr_x_test = x_test.to_numpy()

predictions = clf.predict(arr_x_test)
#file for submission
solutions = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': predictions})
solutions.head(5)
solutions.to_csv('sample_submission_v1.csv', index=False)
#data exploratory analysis: some plots
import matplotlib.pyplot as plt

result = pd.concat([x_train, y_train], axis=1)

plt.matshow(result.corr())
plt.yticks(range(result.shape[1]), result.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=12);

plt.show()
#result.plot(kind='bar',x='Age',y='Survived==0',color='red')
#result.groupby('Sex_cat')['Survived'].plot(kind='bar')
not_survived = result['Survived'] == 0
survived = result['Survived'] == 1
result[not_survived]['Age'].plot(kind='hist',bins=[0,20,40,60,80,100],rwidth=0.8, alpha=0.5, label='Not surv.', legend=True)
result[survived]['Age'].plot(kind='hist',bins=[0,20,40,60,80,100],rwidth=0.8, alpha=0.5, label='Surv.', legend=True)

plt.show()
result[not_survived]['Sex_cat'].plot(kind='hist',bins=[0,0.5,1],rwidth=0.8, alpha=0.5, label='Not surv.', legend=True)
result[survived]['Sex_cat'].plot(kind='hist',bins=[0,0.5,1],rwidth=0.8, alpha=0.5, label='Surv.', legend=True)
plt.show()
result[not_survived]['Embarked_cat'].plot(kind='hist',bins=[0,0.5,1,1.5,2,2.5,3,3.5],rwidth=0.8, alpha=0.5, label='Not surv.', legend=True, logy=True)
result[survived]['Embarked_cat'].plot(kind='hist',bins=[0,0.5,1,1.5,2,2.5,3,3.5],rwidth=0.8, alpha=0.5, label='Surv.', legend=True, logy=True)
plt.show()
result[not_survived]['Parch'].plot(kind='hist',bins=[0,0.5,1,1.5,2,2.5,3,3.5],rwidth=0.8, alpha=0.5, label='Not surv.', legend=True, logy=True)
result[survived]['Parch'].plot(kind='hist',bins=[0,0.5,1,1.5,2,2.5,3,3.5],rwidth=0.8, alpha=0.5, label='Surv.', legend=True, logy=True)
plt.show()
result[not_survived]['SibSp'].plot(kind='hist',bins=[0,0.5,1,1.5,2,2.5,3,3.5],rwidth=0.8, alpha=0.5, label='Not surv.', legend=True, logy=True)
result[survived]['SibSp'].plot(kind='hist',bins=[0,0.5,1,1.5,2,2.5,3,3.5],rwidth=0.8, alpha=0.5, label='Surv.', legend=True, logy=True)
plt.show()
result[not_survived]['Pclass'].plot(kind='hist',bins=[0,0.5,1,1.5,2,2.5,3,3.5],rwidth=0.8, alpha=0.5, label='Not surv.', legend=True, logy=True)
result[survived]['Pclass'].plot(kind='hist',bins=[0,0.5,1,1.5,2,2.5,3,3.5],rwidth=0.8, alpha=0.5, label='Surv.', legend=True, logy=True)
plt.show()
result[not_survived]['Fare'].plot(kind='hist',bins=[0,20,40,60,80,100,125,150,175,200],rwidth=0.8, alpha=0.5, label='Not surv.', legend=True, logy=True)
result[survived]['Fare'].plot(kind='hist',bins=[0,20,40,60,80,100,125,150,175,200],rwidth=0.8, alpha=0.5, label='Surv.', legend=True, logy=True)
plt.show()