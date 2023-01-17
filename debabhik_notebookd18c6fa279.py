# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

test_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
train_data['Survived'].where((train_data['Survived']==1.0)).count()
train_data.info()

test_data.info()
plt.scatter(train_data.index.tolist(),train_data['Fare'].tolist())
sns.barplot('Survived','Fare',data=train_data)
fare_max = train_data['Fare'].max()

train_data['Fare'] = train_data['Fare']/fare_max
sns.pairplot(train_data[['Survived','Fare','Pclass','Sex','SibSp','Parch','Embarked']], hue='Pclass')
#fill missing data

train_data['Age'].fillna(np.median(train_data['Age'].dropna()),inplace=True)

test_data['Age'].fillna(np.median(test_data['Age'].dropna()),inplace=True)

train_data['Embarked'].fillna('S',inplace=True)

test_data['Fare'].fillna(np.median(test_data['Fare'].dropna()),inplace=True)
# map data

train_data['Embarked'].replace({'S':1,'C':2,'Q':3},inplace=True) ##S-1,C-2,Q-3

train_data['Sex'].replace({"male":1,"female":2},inplace=True)  ##male-1,female-2

test_data['Embarked'].replace({'S':1,'C':2,'Q':3},inplace=True) ##S-1,C-2,Q-3

test_data['Sex'].replace({"male":1,"female":2},inplace=True)  ##male-1,female-2
train_data[['Age','Fare']]=train_data[['Age','Fare']].astype(int)

test_data[['Age','Fare']]=test_data[['Age','Fare']].astype(int)
test_data
train_data
lr = LogisticRegression(C=1.0,random_state=0,tol=0.0001)
x_train,x_test,y_train,y_test = train_test_split(train_data.ix[:,1:],train_data['Survived'],test_size=0.25,random_state=0)
clf = lr.fit(x_train,y_train)
clf.score(x_test,y_test)
pr = lr.predict(x_test)
pr
y_test
accuracy_score(y_test,pr)