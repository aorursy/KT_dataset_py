# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/train.csv") 
dataset.head()
dataset[['class1','class2']] = pd.get_dummies(dataset['Pclass'],drop_first=True)
dataset.head()
import seaborn as sns
dataset.notnull()
sns.heatmap(dataset.notnull())

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age,axis = 1)
sns.heatmap(dataset.isnull())
dataset.drop('Cabin',inplace = True,axis = 1)
dataset['Sex'] = pd.get_dummies(dataset['Sex'],drop_first=True) 
dataset.head()
embark = pd.get_dummies(dataset['Embarked'],drop_first = True)

dataset.head()
dataset.dropna(inplace = True)
sns.heatmap(dataset.isnull())
dataset = pd.concat([dataset,embark],axis = 1)
dataset.head()
dataset.drop(['PassengerId','Name','Ticket','Embarked'],axis = 1,inplace = True)
dataset.head()
x = dataset.drop('Survived',axis = 1)
x = x.drop('Pclass',axis = 1)
y = dataset['Survived']
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 300,criterion = 'entropy')
x.dropna(inplace = True)
y.dropna(inplace = True)
rfc.fit(x,y)
test = pd.read_csv("../input/test.csv")
test.head()
embark = pd.get_dummies(test['Embarked'],drop_first=True)
sex = pd.get_dummies(test['Sex'],drop_first=True)
test.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],inplace = True,axis = 1)
new_test = pd.concat([test,sex,embark],axis = 1)
new_test.head()
new_test['Age'] = new_test[['Age','Pclass']].apply(impute_age,axis = 1)
sns.heatmap(new_test.isnull())
new_test['Sex'] = new_test['male']
new_test.drop('male',axis = 1,inplace = True)
new_test.head()
x.head()
new_test[['class1','class2']] = pd.get_dummies(new_test['Pclass'],drop_first = True)
new_test.head()
new_test = new_test.drop('Pclass',axis = 1)
sns.heatmap(new_test.notna())
x.head()
test_set = new_test[['Sex','Age','SibSp','Parch','Fare','class1','class2','Q','S']]
sns.heatmap(test_set.isna())
np.where(test_set.isna())
new_test = test_set.fillna(test_set['Fare'].mean())
y_pred = rfc.predict(new_test)
y_pred
test_1 = pd.read_csv("../input/test.csv")
test_1.head()
submission = pd.DataFrame({"PassengerID":test_1['PassengerId'],"Survived":y_pred})
submission.PassengerID = submission.PassengerID.apply(int)
submission.Survived = submission.Survived.apply(int)
submission.head()
submission.groupby('Survived').count()
#CSV file for submission
submission.to_csv("titanic_submission_rfc.csv", index=False)

