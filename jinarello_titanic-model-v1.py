import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



train['Survived'] = train['Survived'].astype(int)



data = train.append(test, ignore_index=True)



passengerId = test.PassengerId



train_idx = len(train)

test_idx = len(data) - len(test)
data.head(10)

#name, sex, cabin, embarked are objects, rest int or float
#maybe people with longer names are higher class, or as female spouses or children who

#had their tickets bought for them were more likely to live.

def count_letters(x):

    return len(str(x))



data['Name'] = data['Name'].apply(count_letters)



#according to regplot it is more likely
import seaborn as sb

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

sb.regplot(data=data, x="Name", y="Survived")

def binary(x):

    if x == "male":

        return 0

    else:

        return 1

        

data['Sex'] = data['Sex'].apply(binary)
#NaN finder

nans = lambda df: df[df.isnull().any(axis=1)]

nans(train)
data['Age'].fillna(train['Age'].median(), inplace=True)

data = data.drop(['Cabin', 'Ticket'], axis=1)

data['Embarked'] = data['Embarked'].fillna('N', axis=0)
data.dtypes
data['Family'] = data.Parch + data.SibSp

embarked_dummies = pd.get_dummies(data.Embarked, prefix="Embarked")

data_dummies = pd.concat([data, embarked_dummies], axis=1)



data_dummies.head()
ftrain = data_dummies[0:train_idx]

ftest = data_dummies[test_idx:]



features = ['Age', 'Fare', 'Name', 'Pclass', 'Sex', 'Family', 'Embarked_C', 'Embarked_N', 'Embarked_Q', 'Embarked_S']

y = ftrain.Survived.astype(int)

X = ftrain[features]



#y_test = ftest.Survived

X_test = ftest[features]
nans = lambda df: df[df.isnull().any(axis=1)]

nans(ftrain)
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score



XG_model = XGBRegressor(n_estimators=500, learning_rate=0.1, random_state=1)

XG_model.fit(X, y, verbose=False)



XG_model_prediction = XG_model.predict(X_test).astype(int)



# XG_model_train_mae = mean_absolute_error(XG_model_test_prediction, y_test)

# XG_model_R2 = r2_score(XG_model_test_prediction, y_test)



# print("Validation MAE for XGBoost: {}".format(XG_model_test_mae))

# print("R^2 score for XG: {}".format(XG_model_R2))
submission = pd.DataFrame({'PassengerId': passengerId, 'Survived': XG_model_prediction})

submission.to_csv('submissionv1.csv', index=False)



print(submission)