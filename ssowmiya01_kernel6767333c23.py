# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

# Any results you write to the current directory are saved as output.
data1=pd.read_csv('../input/titanic/train.csv')

data2=pd.read_csv('../input/titanic/test.csv')

data3=pd.read_csv('../input/titanic/gender_submission.csv')
print(data1.isnull().sum())

data2.isnull().sum()
data1.drop(['Name'],axis=1,inplace=True)

data2.drop(['Ticket'],axis=1,inplace=True)

data2.drop(['Name'],axis=1,inplace=True)

data1.drop(['Ticket'],axis=1,inplace=True)

data1.head(6)
data1['Age']=data1['Age'].fillna(data1['Age'].mode()[0])

data2['Age']=data2['Age'].fillna(data2['Age'].mode()[0])
my_imputer=SimpleImputer(strategy='most_frequent')

imptrain=pd.DataFrame(my_imputer.fit_transform(data1))

imptest=pd.DataFrame(my_imputer.fit_transform(data2))

imptrain.columns=data1.columns

imptest.columns=data2.columns

data1=imptrain.copy()

data2=imptest.copy()
y=data1['Survived']

data1.drop(['Survived'],axis=1,inplace=True)
encoder_mod=LabelEncoder()

ltrain=data1.copy()

ltest=data2.copy()

s=(data1.dtypes=='object')

obj_col=list(s[s].index)
for col in obj_col:

    ltrain[col]=encoder_mod.fit_transform(data1[col].astype(str))

    ltest[col]=encoder_mod.fit_transform(data2[col].astype(str))
xtrain,xvalid,ytrain,yvalid=train_test_split(ltrain, y, test_size=0.33, random_state=42)
sns.scatterplot(x=xtrain['PassengerId'],y=ytrain)
ytrain=ytrain.astype('int') 

logreg=LogisticRegression()

logreg.fit(xtrain,ytrain)
yvalid=yvalid.astype('int')

y_pred = logreg.predict(xvalid)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(xvalid, yvalid)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(yvalid, y_pred)

print(confusion_matrix)

from sklearn.metrics import classification_report

print(classification_report(yvalid, y_pred))
ytest=logreg.predict(ltest)
id =np.array(ltest["PassengerId"]).astype(int)

my_solution = pd.DataFrame(ytest,id, columns = ["Survived"])

print(my_solution)
my_solution.to_csv("solution_one.csv", index_label = ["PassengerId"])