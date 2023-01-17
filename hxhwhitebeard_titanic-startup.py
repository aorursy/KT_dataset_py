# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname," ",filenames)
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
 
train= pd.read_csv('/kaggle/input/titanic/train.csv')
test= pd.read_csv('/kaggle/input/titanic/test.csv')
print(train.shape)
selectedFeature=["Survived",'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train=train[selectedFeature]
test=test[[ 'PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
 
print('train before \n',train.info())
train['Age'].fillna(29,inplace=True)
train['Embarked'].fillna('S',inplace=True)
print('train after \n',train.info())
print("-------test---------")
print('test before\n',test.info())
test['Age'].fillna(29,inplace=True)
test['Fare'].fillna(35.6,inplace=True)

print('test after\n',test.info())
 
# sex and emberk -> to numeric
train['Sex']=train['Sex'].astype('category')
train['Embarked']=train['Embarked'].astype('category')
train.dtypes
train["Sex_cat"] = train["Sex"].cat.codes
train["Embarked_cat"]=train["Embarked"].cat.codes
train.head()
#for test set
test['Sex']     =test['Sex']     .astype('category')
test['Embarked']=test['Embarked'].astype('category')
test["Sex_cat"] = test["Sex"].cat.codes
test["Embarked_cat"]=test["Embarked"].cat.codes
test.head()
X=train[['Pclass','Sex_cat','Age','SibSp','Parch','Fare','Embarked_cat']]
print(X.dtypes)
y=train['Survived']
X=np.asarray(X)
print(X[0:5])
y=y.values
y[0:5]
# for test set
Xtest=test[[ 'Pclass','Sex_cat','Age','SibSp','Parch','Fare','Embarked_cat']]
print(Xtest.dtypes)
 
Xtest=np.asarray(Xtest)
print(Xtest[0:5])
 
 
from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)
Xtest=preprocessing.StandardScaler().fit(Xtest).transform(Xtest)
 
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=0.01,solver='liblinear').fit(X,y)
lr
from sklearn.metrics import jaccard_score
yhat=lr.predict(Xtest )
yhat.shape

 
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': yhat})
my_submission.to_csv('submission.csv', index=False)
output= pd.read_csv('/kaggle/working/submission.csv')
output.shape
