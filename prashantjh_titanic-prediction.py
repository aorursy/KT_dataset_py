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
df_train=pd.read_csv('/kaggle/input/titanic/train.csv')
df_train
df_train.isnull()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df_train)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=df_train)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=df_train)
sns.distplot(df_train['Age'].dropna(),kde=False)
sns.countplot(x='SibSp',data=df_train)
sns.set_style('whitegrid')
sns.boxenplot(x='Pclass',y='Age',data=df_train,palette='winter')
def calculate_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
df_train['Age'] = df_train[['Age','Pclass']].apply(calculate_age,axis=1)
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_train.drop('Cabin',inplace=True,axis=1)
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)

df_train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df_train.head()
train=pd.concat([df_train,sex,embark],axis=1)
train.drop('Survived',axis=1).head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500)
rnd_clf.fit(X_train,y_train)
y_pred=rnd_clf.predict(X_test)
from sklearn import metrics

metrics.accuracy_score(y_test,y_pred)
metrics.confusion_matrix(y_test,y_pred)