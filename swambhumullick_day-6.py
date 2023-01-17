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
import pandas as pd

train = pd.read_csv("../input/titanicdataset-traincsv/train.csv")
train
train.head()
train.count()
train[train['Sex'].str.match("male")].count()
train[train['Sex'].str.match("female")].count()
import seaborn as sns

sns.countplot(x='Survived',hue='Pclass',data=train)
o=pd.DataFrame(columns=['sd'])

o['sd']=train['Sex']

o[o['sd'].str.match("female")].count()
sns.countplot(x='Survived',hue='Sex',data=train)
pclass = pd.get_dummies(train['Pclass'])

embark = pd.get_dummies(train['Embarked'])

sex = pd.get_dummies(train['Sex'])
train
train.drop(["PassengerId","Pclass","Name","Sex","Ticket","Embarked","Cabin","Age"],axis=1,inplace=True)

train
train = pd.concat([train,pclass,sex,embark],axis=1)
X=train.drop("Survived",axis=1)

Y=train["Survived"]
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3 , random_state= 101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

train
logmodel.fit(X_train,Y_train)
predictions=logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(Y_test,predictions))
from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test,predictions)
from sklearn.metrics import accuracy_score

print('accuracy score :',accuracy_score(Y_test,predictions))