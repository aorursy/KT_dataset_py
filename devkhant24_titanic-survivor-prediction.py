# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import xgboost

import scipy

import pickle

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import chi2

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
dum = pd.get_dummies(df[['Sex','Embarked']],drop_first=True)

df.drop(['Sex','Embarked'],axis=1,inplace=True)

df = pd.concat([dum,df],axis=1)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df = df[['Sex_male','Embarked_S','Pclass','Age','SibSp','Parch','Fare','Survived']]
dft = pd.read_csv('/kaggle/input/titanic/test.csv')
dft.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
dum = pd.get_dummies(dft[['Sex','Embarked']],drop_first=True)

dft.drop(['Sex','Embarked'],axis=1,inplace=True)

dft = pd.concat([dum,dft],axis=1)
dft['Age'] = dft['Age'].fillna(dft['Age'].mean())

dft['Fare'] = dft['Fare'].fillna(dft['Fare'].mean())
dft = dft[['Sex_male','Embarked_S','Pclass','Age','SibSp','Parch','Fare']]
x = df.drop(['Survived'],axis=1)

y = df['Survived']
bestfeature = SelectKBest(score_func=chi2,k=7)

fit = bestfeature.fit(x,y)

dfscore = pd.DataFrame(fit.scores_)

dfcolumn = pd.DataFrame(x.columns)

features = pd.concat([dfcolumn,dfscore],axis=1)

features.columns = ['specs','score']

print(features.nlargest(10,'score'))
#z = np.abs(scipy.stats.zscore(df))

#df=df[(z<3).all(axis=1)]
lg = LogisticRegression()

xg = xgboost.XGBClassifier()

dt = DecisionTreeClassifier(random_state=1)

rf = RandomForestClassifier(random_state=1)

nb = GaussianNB()

svm = SVC(kernel='linear')
#score = cross_val_score(rf,x,y,cv=5)

#score.mean()
xg.fit(x,y)
pickle.dump(xg,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
yp = xg.predict(dft)

pred = pd.DataFrame(yp)

pred.columns = ['Survived']

d = pd.concat([dft,pred],axis=1)
train = pd.concat([df,d],axis=0,sort=False)
X = train.drop(['Survived'],axis=1)

Y = train['Survived']
xg.fit(X,Y)
ypred = xg.predict(dft)
a = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission = pd.DataFrame({'PassengerId':a['PassengerId'],'Survived':ypred})
submission.to_csv('gender_submission.csv',index=False)