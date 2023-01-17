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
import pandas as pd 

train=pd.read_csv('/kaggle/input/titanic/train.csv')

print(train.head())



output=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train.columns
test=pd.read_csv('/kaggle/input/titanic/test.csv')

print(test.head())

test.columns
print(train.shape)

print(test.shape)

x_axis=train['Ticket']

y_axis=train['PassengerId']

import matplotlib.pyplot as plt 

fig=plt.figure()

ax=fig.add_subplot(1,1,1)

ax.scatter(x_axis,y_axis)

plt.title('plot')

plt.xlabel('ticket')

plt.ylabel('survuved')

print(train.isna().sum())

print(test.isna().sum())

data = pd.concat([train, test], sort = False)



data.head()


#Fill Missing numbers with median

data['Age'] = data['Age'].fillna(value=data['Age'].median())

data['Fare'] = data['Fare'].fillna(value=data['Fare'].median())

data.info()

data['Embarked'] = data['Embarked'].fillna('S')

data['Cabin'] = data['Cabin'].fillna('NAN')

data['Cabin']=data['Cabin'].replace('NAN','C205')

data.info()

print(data.isna().sum())

print(data['Cabin'])
# binarizing sex column

data.Sex[data.Sex == 'male'] = 1

data.Sex[data.Sex == 'female'] = 0



print(data['Sex'])
# names column titles

import re

def get_title(name):

    title_search = re.search('([A-Za-z]+\.)', name)

    

    if title_search:

        return title_search.group(1)

    return ""

data['Names'] = data['Name'].apply(get_title)

data['Names'].value_counts()



print(data['Names'])
#Age

data.loc[ data['Age'] <= 20, 'Age'] = 0

data.loc[(data['Age'] > 20) & (data['Age'] <= 40), 'Age'] = 1

data.loc[ data['Age'] > 40, 'Age'] = 2

print(data['Age'].head(100))
print(data.head)

# survived column

rand=np.random.randint(0,1)

data['Survived']=data['Survived'].replace(np.nan,rand)

print(data['Survived'])
import pandas as pd



data=pd.get_dummies(data)



# making model



X=data.drop("Survived",axis=1)

y=data['Survived']





from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1)



# Logistic Regression 



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

pred=[]

log=LogisticRegression(C=0.5,penalty='l2',l1_ratio=0)

res=log.fit(X_train,y_train.values.ravel())

pred=log.predict(X_test)

print(accuracy_score(pred,y_test))



print('confusion matrix')

print(confusion_matrix(y_test, pred))



print(classification_report(y_test, pred))
from sklearn.svm import SVC



# Support vector

svc=SVC()



model=svc.fit(X_train,y_train)

pred=svc.predict(X_test)



svc_score=svc.score(X_train,y_train)



print(svc_score)

print(confusion_matrix(y_test, pred))

from sklearn.ensemble import RandomForestClassifier



rfc=RandomForestClassifier(n_estimators=80)

rfc.fit(X_train,y_train)



pred=rfc.predict(X_test)

Score=rfc.score(X_test,y_test)

print(Score)

print(confusion_matrix(y_test, pred))