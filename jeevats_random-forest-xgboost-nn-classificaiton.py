# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from sklearn import preprocessing
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv('/kaggle/input/titanic/train.csv')
df_test=pd.read_csv('/kaggle/input/titanic/test.csv')
df_sub=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_sub.head()

print(df_train.head())
df_test.head()
df_sub.head()
df_train.columns
df_train['Sex']=df_train['Sex'].fillna('0')
df_train['Cabin']=df_train['Cabin'].fillna('0')
df_train['Embarked']=df_train['Embarked'].fillna('0')
df_train['Pclass']=df_train['Pclass'].fillna(0)
df_train['Age']=df_train['Age'].fillna(0)
df_train['SibSp']=df_train['SibSp'].fillna(0)
df_train['Parch']=df_train['Parch'].fillna(0)
df_train['Fare']=df_train['Fare'].fillna(0)

df_test['Sex']=df_test['Sex'].fillna('0')
df_test['Cabin']=df_test['Cabin'].fillna('0')
df_test['Embarked']=df_test['Embarked'].fillna('0')
df_test['Pclass']=df_test['Pclass'].fillna(0)
df_test['Age']=df_test['Age'].fillna(0)
df_test['SibSp']=df_test['SibSp'].fillna(0)
df_test['Parch']=df_test['Parch'].fillna(0)
df_test['Fare']=df_test['Fare'].fillna(0)

sex=list(df_train['Sex'])
sex.extend(df_test['Sex'])
cabin=list(df_train['Cabin'])
cabin.extend(df_test['Cabin'])
embarked=list(df_train['Embarked'])
embarked.extend(df_test['Embarked'])
sle=preprocessing.LabelEncoder()
cle=preprocessing.LabelEncoder()
ele=preprocessing.LabelEncoder()
sle.fit(sex)
cle.fit(cabin)
ele.fit(embarked)
df_train['Sex']=sle.transform(df_train['Sex'])
df_train['Cabin']=cle.transform(df_train['Cabin'])
df_train['Embarked']=ele.transform(df_train['Embarked'])

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
X=df_train[['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked']]
Y=df_train.iloc[:,1]
corrmat=df_train.corr()
top_corr=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df_train[top_corr].corr(),annot=True,cmap='RdYlGn')
Y.sample(100)
df_train.head()
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
model=RandomForestClassifier()
model.fit(X,Y)
X_test=df_test[['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked']]

X_test['Sex']=sle.transform(X_test['Sex'])
X_test['Cabin']=cle.transform(X_test['Cabin'])
X_test['Embarked']=ele.transform(X_test['Embarked'])
X_test.head()

prediction=model.predict(X_test)

df_sub=pd.DataFrame()
df_sub['PassengerId']=df_test['PassengerId']
df_sub['Survived']=list(prediction)
df_sub.to_csv('Submission.csv',index=False)
model=xgb.XGBClassifier(n_estimaters=2000,max_depth=8,objective='multi:softprob',seed=0,nthread=-1,learning_rate=0.15,num_class=2,scale_pos_weight=(len(X)/584))
model.fit(X,Y)
X_test=df_test[['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked']]
X_test['Sex']=sle.transform(X_test['Sex'])
X_test['Cabin']=cle.transform(X_test['Cabin'])
X_test['Embarked']=ele.transform(X_test['Embarked'])
prediction=model.predict(X_test)
df_sub=pd.DataFrame()
df_sub['PassengerId']=df_test['PassengerId']
df_sub['Survived']=list(prediction)
df_sub.to_csv('Submission.csv',index=False)
!pip install --upgrade keras

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import metrics
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from keras.metrics import mean_squared_error
from math import sqrt

classifier=Sequential()
classifier.add(Dense(units=100,kernel_initializer='normal',activation='relu',input_dim=X.shape[1]))
classifier.add(Dropout(rate=0.2))


classifier.add(Dense(units=50,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate=0.2))

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


classifier.fit(X,Y,100,100)
X_test=df_test[['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked']]
X_test['Sex']=sle.transform(X_test['Sex'])
X_test['Cabin']=cle.transform(X_test['Cabin'])
X_test['Embarked']=ele.transform(X_test['Embarked'])
prediction=classifier.predict(X_test)
y_pred=[]
for i in prediction:
    if i>=0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
df_sub=pd.DataFrame()
df_sub['PassengerId']=df_test['PassengerId']
df_sub['Survived']=list(y_pred)
df_sub.to_csv('Submission.csv',index=False)
