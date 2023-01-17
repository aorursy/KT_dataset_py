# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras import layers
scrstand = StandardScaler()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_data = pd.read_csv('/kaggle/input/titanic/train.csv')
def pre_process(raw_data):
    raw_data['Sex'].replace({'male':0,'female':1}, inplace=True)
    raw_data.Age = scrstand.fit_transform(raw_data.Age.values.reshape(-1,1))
    raw_data.Fare = scrstand.fit_transform(raw_data.Fare.values.reshape(-1,1))
    dummies = pd.get_dummies(raw_data['Embarked'])
    dummies_par = pd.get_dummies(raw_data['Parch'],prefix='P')
    dumm_new = pd.get_dummies(raw_data['SibSp'],prefix='S')
    dummies_P  = pd.get_dummies(raw_data['Pclass'],prefix='Pr')
    new_data = pd.concat([raw_data,dummies,dumm_new,dummies_P,dummies_par],axis=1)
    new_data.drop(['Embarked','Fare','Pclass','SibSp','Cabin','Ticket','Name','PassengerId'], axis=1,inplace=True)
    return new_data
    
data = pre_process(raw_data)
data.columns
data.head(5)
data['Parch'].value_counts()
data.isnull().sum()
sns.heatmap(data.corr())
plt.subplot(1,3,1)
sns.violinplot(x="Survived", y="Age", data=data)
plt.subplot(1,3,2)
sns.boxplot(x="Survived", y="Age", data=data)
plt.subplot(1,3,3)
sns.catplot(x="Survived", y="Age", data=data)
pd.crosstab(raw_data['Survived'],raw_data['Pclass']).plot(kind='bar')
pd.crosstab(data['Survived'],data['Sex']).plot(kind='bar')
sns.distplot(data['Age'])
pd.crosstab(data['Survived'],data['Q']).plot(kind='bar')
pd.crosstab(data['Survived'],data['S']).plot(kind='bar')
data["Age"].fillna(np.mean(data.Age), inplace = True) 
data.drop('Survived', axis=1,inplace=True)
X = data
Y = raw_data['Survived']
x_train, x_test, y_train,y_test = train_test_split(X,Y, test_size=0.3)
clf = RandomForestClassifier(n_estimators=120)
clf.fit(x_train,y_train)
log = LogisticRegression()
log.fit(x_train,y_train)
ada = AdaBoostClassifier(n_estimators=130, learning_rate=1)
ada.fit(x_train,y_train)
cld_sv  =  svm.SVC()
cld_sv.fit(x_train,y_train)
for i in [clf,log,ada,cld_sv]:
    y_pred = i.predict(x_test)
    print(accuracy_score(y_test,y_pred))
# Keras Model

model_dense = Sequential()
model_dense.add(layers.Dense(10, input_dim=x_train.shape[1],activation='relu'))
model_dense.add(layers.Dropout(0.2))
model_dense.add(layers.Dense(1,activation= 'sigmoid'))
model_dense.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_dense.fit(x_train,y_train, batch_size=100, epochs=30)
model_dense.evaluate(x_test,y_test)