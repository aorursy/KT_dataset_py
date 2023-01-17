import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import accuracy_score

# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")

# fill missing values
meanAge = train.Age.mean()
train.Age.fillna(meanAge, inplace=True)
test.Age.fillna(meanAge, inplace=True)

# feature creation
train['is_female'] = train.Sex.apply(lambda sex: 1 if sex == 'female' else 0)
test['is_female'] = test.Sex.apply(lambda sex: 1 if sex == 'female' else 0)

train = train[['Age', 'is_female', 'Survived']]

x_train = train.drop('Survived', axis=1)
y_train = train.Survived
x_test = test[x_train.columns]

model = lgb.LGBMRegressor()

model.fit(x_train, y_train)

train_pred = model.predict(x_train)
train_pred = np.round(train_pred).astype(int)

print('accuracy', accuracy_score(y_train, train_pred))

test_pred = model.predict(x_test)
test_pred = np.round(test_pred).astype(int)

predicted = pd.DataFrame({
    "PassengerId": test.PassengerId,
    "Survived": test_pred
})

predicted.to_csv('submission.csv', index=False)
