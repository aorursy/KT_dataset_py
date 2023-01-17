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
import re

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn import metrics
path = r"./titanic"



train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train.head()

train.isnull().sum()
pat = re.compile('[\w]+, ([\w]+)')# 이름에서 앞단어 빼내기

def find_name_title(x):

    return pat.findall(x)[0]



#이름 변경

train['Name'] = train['Name'].apply(find_name_title)

test['Name'] = test['Name'].apply(find_name_title)
train['Cabin_yes'] = np.where(train['Cabin'].isnull()==False,1,0)

test['Cabin_yes'] = np.where(test['Cabin'].isnull()==False,1,0)
train
pat_cabin = re.compile('([\w])[0-9]*')



def Cabin_word(x):

    if x is not np.nan and x is not None:

        return pat_cabin.findall(x)[0]

    else: return 'X'



train['Cabin_C'] = train['Cabin'].apply(Cabin_word)

test['Cabin_C'] = test['Cabin'].apply(Cabin_word)
train_dummy = pd.get_dummies(data=train,columns=['Name','Sex','Embarked','Cabin_C']).drop(['PassengerId','Ticket','Cabin'], axis=1)

test_dummy = pd.get_dummies(data=test,columns=['Name','Sex','Embarked','Cabin_C']).drop(['PassengerId','Ticket','Cabin'], axis=1)
knn = KNeighborsRegressor()

# 나이가 있는 데이터로 fit해서 모델을 생성

knn.fit(train_dummy[train_dummy['Age'].isnull()==False][train_dummy.columns.drop('Age')],

       train_dummy[train_dummy['Age'].isnull()==False]['Age'])

guesses = knn.predict(train_dummy[train_dummy['Age'].isnull()==True][train_dummy.columns.drop('Age')])

guesses
train_dummy.loc[train_dummy['Age'].isnull()==True,'Age'] = guesses
train_dummy.isnull().sum()
test_dummy['Age']

train_dummy['Age_cat'] = pd.cut(train_dummy['Age'],10, labels=[*range(10)])

test_dummy['Age_cat'] = pd.cut(test_dummy['Age'],10, labels=[*range(10)])
train_dummy['Fare']
train_dummy['Fare_cat'] = pd.qcut(train_dummy.Fare,5, labels=[*range(5)])

test_dummy['Fare_cat'] = pd.qcut(test_dummy.Fare,5, labels=[*range(5)])
train_dummy['Fare_cat'].value_counts()
train_final = pd.get_dummies(train_dummy, columns=['Age_cat','Fare_cat','Pclass']).drop(['Age','Fare','Pclass'],axis=1, errors='ignore')

test_final = pd.get_dummies(test_dummy, columns=['Age_cat','Fare_cat','Pclass']).drop(['Age','Fare','Pclass'],axis=1, errors='ignore')

train_final
X_train, X_test, y_train, y_test = train_test_split(train_final.iloc[:,1:],train_final['Survived'],test_size=0.2,random_state=42)



rf_clf = RandomForestClassifier()

model = rf_clf.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy_score(pred, y_test)

from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")