import numpy as np

import pandas as pd



train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

submission = pd.read_csv("../input/titanic/gender_submission.csv")



data = pd.concat([train, test], sort=False)
data.head()
data['Sex'].replace(['male','female'], [0, 1], inplace=True)



delete_columns = ['Name', 'PassengerId','Ticket', 'Cabin', 'Embarked']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
X_train.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
categorical_features = ['Pclass', 'Sex']
import lightgbm as lgb





lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)



params = {

    'objective': 'binary'

}



model = lgb.train(

    params, lgb_train,

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=10,

    num_boost_round=1000,

    early_stopping_rounds=10

)



y_pred = model.predict(X_test, num_iteration=model.best_iteration)

y_pred = (y_pred > 0.5).astype(int)
submission['Survived'] = y_pred

submission.to_csv("submission_lightgbm.csv", index=False)



submission.head()