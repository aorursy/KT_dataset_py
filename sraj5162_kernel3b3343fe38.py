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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
women = train_data.loc[train_data.Sex=='female']['Survived']
rate_women = sum(women)/len(women)
print("% of women who survived",rate_women)
men = train_data.loc[train_data.Sex=="male"]['Survived']
rate_men = sum(men)/len(men)
print("% of men survived", rate_men)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
def mean_age(lst):
    age = lst[0]
    Class= lst[1]
    if pd.isnull(age):
        if Class == 1:
            return 37
        elif Class ==2:
            return 29
        elif Class ==3:
            return 24
    else:
        return age
    
train_data['Age'] = train_data[['Age','Pclass']].apply(mean_age,axis=1)
test_data['Age'] = test_data[['Age','Pclass']].apply(mean_age,axis=1)
train_data.isnull().sum()
test_data.isnull().sum()
train_data=train_data.drop('Cabin',axis=1)
test_data=test_data.drop('Cabin',axis=1)

train_data.dropna(inplace =True)
test_data.dropna(inplace =True)
sex_dumm_tra = pd.get_dummies(train_data['Sex'],drop_first=True)
embark_dumm_tra = pd.get_dummies(train_data['Embarked'],drop_first=True)
sex_dumm_tes = pd.get_dummies(test_data['Sex'],drop_first=True)
embark_dumm_tes = pd.get_dummies(test_data['Embarked'],drop_first=True)

train_data=train_data.drop(['Sex','Embarked'],axis=1)
test_data=test_data.drop(['Sex','Embarked'],axis=1)
train_data = pd.concat([train_data,sex_dumm_tra,embark_dumm_tra],axis=1)
test_data = pd.concat([test_data,sex_dumm_tes,embark_dumm_tes],axis=1)
train_data = train_data.drop('Name',axis=1)
test_data = test_data.drop('Name',axis=1)
train_data.head()
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=1000)

train_data = train_data.drop('Ticket',axis=1)
test_data = test_data.drop('Ticket',axis=1)

X = train_data.drop('Survived',axis=1)
y = train_data['Survived']

#model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

logmodel.fit(X, y)
predictions = logmodel.predict(test_data)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
