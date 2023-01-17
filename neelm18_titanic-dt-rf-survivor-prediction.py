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
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

train_data.head()
from sklearn.preprocessing.label import LabelEncoder

#le_Fare = le_Sex  = LabelEncoder()



#train_data['Sex'] = le_Sex.fit_transform(train_data['Sex'])

test_data.head(25)



#print(train_data['Fare'].unique())
#print(train_data['Fare'].max())

#print(pd.cut(train_data['Fare'], 10))

train_data['binned_Fare'] = pd.cut(train_data['Fare'], 10)

#test_data['binned_Fare'] = pd.cut(test_data['Fare'], 10)



#print(train_data['binned_Fare'].unique())

print(test_data.Fare.dtype)

#test_data.head(50)

#test_data['Fare'] = test_data['Fare'].astype(float)

test_data['binned_Fare'] = test_data['Fare']



test_data.loc[test_data['Fare'] <= 51.233, 'binned_Fare'] = 0

test_data.loc[((test_data['Fare'] > 51.233) & (test_data['Fare'] <= 102.466), 'binned_Fare')] = 1

test_data.loc[(test_data['Fare'] > 102.466) & (test_data['Fare'] <= 153.699), 'binned_Fare'] = 2

test_data.loc[(test_data['Fare'] > 153.699) & (test_data['Fare'] <= 204.932), 'binned_Fare'] = 3

test_data.loc[(test_data['Fare'] > 204.932) & (test_data['Fare'] <= 256.165), 'binned_Fare'] = 4

test_data.loc[(test_data['Fare'] > 256.165) & (test_data['Fare'] <= 307.398), 'binned_Fare'] = 5

test_data.loc[(test_data['Fare'] > 307.398) & (test_data['Fare'] <= 461.096), 'binned_Fare'] = 6

test_data.loc[(test_data['Fare'] > 461.096) & (test_data['Fare'] <= 512.329), 'binned_Fare'] = 7

test_data['binned_Fare'] = test_data['binned_Fare'].fillna(0.0).astype(int)

test_data.head(50)

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



train_data_X = train_data.drop(['PassengerId','Survived','Name','Age','Ticket','Fare','Cabin','Embarked'],axis ='columns')

X_test = test_data.drop(['PassengerId','Name','Age','Ticket','Fare','Cabin','Embarked'],axis ='columns')

le_Fare_test = le_Sex_test = LabelEncoder()



#X_test['binned_Fare'] = le_Fare_test.fit_transform(X_test['binned_Fare'])

X_test['Sex'] = le_Sex_test.fit_transform(X_test['Sex'])





le_Survived = le_Fare = le_Sex = LabelEncoder()

train_data_X['binned_Fare'] = le_Fare.fit_transform(train_data_X['binned_Fare'])

train_data_X['Sex'] = le_Sex.fit_transform(train_data_X['Sex'])

y = train_data['Survived']

#y['Survived'] = le_Survived.fit_transform(y['Survived'])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(train_data_X, y)

predictions = model.predict(X_test)

X_test.head(75)

print(model.score(train_data_X, y))



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.model_selection._validation import cross_val_score



MLModel = DecisionTreeClassifier(criterion="entropy", max_depth=3)

MLModel = MLModel.fit(train_data_X, y)

#K-Fold



score = cross_val_score(MLModel,train_data_X, y,cv=7,scoring='accuracy')

print("Cross Val score ", score)

print("Total Cross Val Score ",round(np.mean(score)*100,2))