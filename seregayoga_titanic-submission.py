import numpy as np

import pandas as pd



from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import Imputer

import xgboost as xgb



import re
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_x = train_data.drop(['PassengerId', 'Survived', 'Ticket'], axis=1)

test_x = test_data.drop(['PassengerId', 'Ticket'], axis=1)



cabins = {

    None if pd.isnull(cabin) else cabin: i

    for i, cabin in enumerate(train_x['Cabin'])

}



def map_age(age):

    if 0 <= age < 3:

        return 0

    if 3 <= age < 17:

        return 1

    if 17 <= age < 30:

        return 2

    if 30 <= age < 40:

        return 3

    if 40 <= age < 50:

        return 4

    if 50 <= age < 60:

        return 5

    if age >= 60:

        return 6

    

def map_name(name):

    m = re.search('(Mr\.|Mrs\.|Miss\.)', name)

    if not m:

        return None

    if m.group(0) == 'Mr.':

        return 0

    if m.group(0) == 'Mrs.':

        return 1

    if m.group(0) == 'Miss.':

        return 2



combined = [train_x, test_x]

for i, x in enumerate(combined):

    combined[i]['Embarked'] = x['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, None: 3})

    combined[i]['Sex'] = x['Sex'].map({'male': 0, 'female': 1})

    combined[i]['Cabin'] = x['Cabin'].map(cabins)

    combined[i]['Age'] = x['Age'].map(map_age, na_action=None)

    combined[i]['Name'] = x['Name'].map(map_name, na_action=None)

    imputer = Imputer(axis=1)

    combined[i] = imputer.fit_transform(x)



train_x = combined[0]

test_x = combined[1]



train_y = train_data["Survived"]

# dtc = DecisionTreeClassifier()

# dtc.fit(train_x, train_y)



# pred_y = dtc.predict(test_x)



gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

gbm.fit(train_x, train_y)



pred_y = gbm.predict(test_x)



submission = pd.DataFrame({

    "PassengerId": test_data["PassengerId"],

    "Survived": pred_y,

})

submission.to_csv('submission.csv', index=False)