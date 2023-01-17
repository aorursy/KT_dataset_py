# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



dt = pd.read_csv('../input/train.csv')

# Any results you write to the current directory are saved as output.
dt.head()
dt.info()

dt.describe()
import seaborn as sns

sns.heatmap(dt.isnull(),cbar=False, cmap='viridis')

dt.columns
sns.boxplot(x='Pclass',y='Age',data=dt)
dt['Age'].fillna(value=dt['Age'].mean(), inplace=True)
dt.head()
dt.drop(['Name','Ticket','Cabin'], inplace=True, axis=1)
dt.head()
sns.pairplot(dt, hue='Survived')
sns.pairplot(kind='reg', data=dt, hue='Survived')
pd.get_dummies(dt['Sex'])
sex = pd.get_dummies(dt['Sex'],drop_first=True)
embark = pd.get_dummies(dt['Embarked'],drop_first=True)
dt = pd.concat([dt,sex,embark],axis=1)
dt.head()
dt.drop(['Sex','Embarked'],axis=1,inplace=True)
dt.head()
x = dt[['PassengerId','Pclass','Age','Parch','Fare','male','Q','S']]

y = dt['Survived']

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y)
from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()

lm.fit(x_train,y_train)
pred = lm.predict(x_test)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,pred))