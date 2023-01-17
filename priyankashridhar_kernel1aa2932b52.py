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

import numpy as np

import seaborn as sn

from matplotlib import pyplot as plt

%matplotlib inline
data=pd.read_csv('/kaggle/input/titanic123/ChanDarren_RaiTaran_Lab2a.csv')

data.head()
data.info()
data.describe()
data.isnull().sum()
import seaborn as sn
data['Survived'].value_counts()

#print((data['Survived']==1)/len(data['PassengerId']))
print((342)/len(data['PassengerId']))   #only 38% of survaival possibility

sn.countplot(data['Survived'])
data['Pclass'].value_counts()
sn.countplot(x='Pclass',data=data,hue='Survived') #first class ppl survived more than other classes
data['Sex'].value_counts()
sn.countplot(data['Sex'])
sn.countplot(x='Sex',data=data,hue='Survived')
data['Age'].plot.hist(bins=20)
data['Fare'].plot.hist(bins=10)
plt.boxplot(x=data['Fare'])
data[data['Fare']>500] 
data['Embarked'].value_counts()
data['Embarked'].value_counts().plot(kind='bar')
sn.boxplot(x="Pclass",y="Age",data=data)
data.head()
data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

data.head()
data.isnull().sum()
##drop missing values
data.dropna(inplace=True)

data.isnull().any()
sex=pd.get_dummies(data['Sex'],drop_first=True)

emb=pd.get_dummies(data['Embarked'],drop_first=True)

pclass=pd.get_dummies(data['Pclass'],drop_first=True)
data=pd.concat([data,sex,emb,pclass],axis=1)

data.head()
data.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)
data.head()
corr=data.corr()

sn.heatmap(corr,annot=True,cmap=sn.diverging_palette(20, 220, n=200))
x=data.drop('Survived',axis=1)

y=data['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
Predictions = model.predict(X_test)

Predictions
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test , Predictions)
from sklearn.metrics import classification_report

classification_report(y_test,Predictions)
from sklearn.metrics import accuracy_score

accuracy_score(y_test , Predictions)

my_submission = pd.DataFrame({'Survived': Predictions})

print(my_submission)