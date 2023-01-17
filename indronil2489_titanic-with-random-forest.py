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
df = pd.read_csv('../input/titanic/train.csv')
id_train = np.array(df['PassengerId'])
y_train = np.array(df['Survived'])
x_train = np.array(df.drop(['PassengerId','Survived','Name','Age','Ticket','Cabin'], axis=1))
df2 = pd.read_csv('../input/titanic/test.csv')
id_test = np.array(df2['PassengerId'])
x_test = np.array(df2.drop(['PassengerId','Name','Age','Ticket','Cabin'], axis=1))
freq =df['Embarked'].dropna().mode()[0]
df['Embarked'] = df['Embarked'].fillna(freq)
x_train[:,5]=df['Embarked']
x_test[:,5]
freq2 =df2['Fare'].dropna().mode()[0]
df2['Fare'] = df2['Fare'].fillna(freq2)
x_test[:,4]=df2['Fare']
x_test[:,4]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(np.unique(x_train[:,5]))
x_train[:,5]=le.transform(x_train[:,5])
x_test[:,5]=le.transform(x_test[:,5])
le.fit(np.unique(x_train[:,1]))
x_train[:,1]=le.transform(x_train[:,1])
x_test[:,1]=le.transform(x_test[:,1])
x_train=x_train.astype(float)
x_test=x_test.astype(float)
from sklearn.ensemble import RandomForestClassifier

rforest = RandomForestClassifier(n_estimators = 1000, random_state = 1) 
rforest.fit(x_train,y_train)
y_test = rforest.predict(x_test)
y_test
y_test.astype(int)
df3 = pd.DataFrame()
df3['PassengerId'] = id_test.reshape(len(y_test)).tolist()
df3['Survived'] = y_test.reshape(len(y_test)).tolist()
df3.to_csv("./file.csv", sep=',',index=True)