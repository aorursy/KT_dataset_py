# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

filename = "../input/titanic-train-dataset/train.csv"
df = pd.read_csv(filename)
df.head(3)
sns.countplot(x='Survived',hue='Sex',data=df)

df.hist('Age')
df.isnull()
dfnull= df.isnull().sum()
dfnull.hist()
sns.heatmap(df.isnull(),xticklabels=False,cbar=False)

sns.boxplot(x='Pclass', y='Age',data=df)
df.head()
dfNew = df.drop('Cabin',axis=1)
dfNew.dropna(inplace=True)
dfNew.isnull().sum()
sns.heatmap(dfNew.isnull(),xticklabels=False,cbar=False)

sex = pd.get_dummies(dfNew['Sex'], drop_first=True)
sex.head()
emb = pd.get_dummies(dfNew['Embarked'], drop_first=True)
emb.head()
pcl = pd.get_dummies(dfNew['Pclass'], drop_first=True)
pcl.head()



dfNew = pd.concat([dfNew,sex,emb,pcl],axis=1)
dfNew.head()
dfNew.drop(['PassengerId','Name','Sex','Ticket','Pclass','Embarked'],axis=1,inplace=True)
dfNew.head()

x = dfNew.drop('Survived',axis=1)
y = dfNew['Survived']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=1)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
predict = model.predict(x_test)
model.score(x_test,y_test)
model.predict_proba(x_test)
from sklearn.metrics import classification_report
classification_report(y_test,predict)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predict)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predict)