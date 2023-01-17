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
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
"""

from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()



features = ["Sex", "Age", "SibSp", "Pclass", "PassengerId"]

#features = ["Sex", "Age", "SibSp","Fare", "Ticket"]

#Figure out how to print data



#print (train_data[0][features])

# "Age", "SibSp", "Sex"; gets a 79

X      = my_imputer.fit_transform (pd.get_dummies(train_data[features]))

X_test = my_imputer.fit_transform (pd.get_dummies(test_data[features]))



#print (train_data.head())



model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index = False)

print("Your submission was successfully saved!")

"""
"""

import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

#features = ["Cabin", "SibSp"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



### test model

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



#model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model = DecisionTreeRegressor (random_state=1)

model.fit(train_X, train_y)

predictions = model.predict (val_X)

val_mae = mean_absolute_error(predictions, val_y)



print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



predictions_float = model.predict(X_test)

predictions = [int(0.5 +x) for x in predictions_float]



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

output.head()

print("Your submission was successfully saved!")

"""


from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()



features = ["Pclass", "Sex", "Fare", "Age"]

X = my_imputer.fit_transform (pd.get_dummies(train_data[features]))

X_test = my_imputer.fit_transform (pd.get_dummies(test_data[features]))



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index = False)

print("Your submission was successfully saved!") 
