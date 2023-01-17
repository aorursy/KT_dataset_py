import pandas as pd

import numpy as np

import xgboost as xgb





people = pd.read_csv('../input/people.csv', nrows=1500000)

submission = pd.read_csv('../input/sample_submission.csv')

train = pd.read_csv('../input/act_train.csv')

test = pd.read_csv('../input/act_test.csv')





y_train = train['outcome']

del train['outcome']

y_train = y_train.as_matrix().ravel()





typed_columns = ['char_' + str(i) for i in range(1,10)] + ['group_1']

for column in typed_columns:

    people[column] = people[column].str.extract('(\d+)', expand=True)

    

people['year'] = people.date.str[:4]

people['month'] = people.date.str[5:7]

people['day'] = people.date.str[8:]

del people['date']





ttyped_columns = ['char_' + str(i) for i in range(1,11)] + ['activity_category']



for column in ttyped_columns:

    train[column] = train[column].str.extract('(\d+)', expand=True)

    

train['year'] = train.date.str[:4]

train['month'] = train.date.str[5:7]

train['day'] = train.date.str[8:]

del train['date']





for column in ttyped_columns:

    test[column] = test[column].str.extract('(\d+)', expand=True)

    

test['year'] = test.date.str[:4]

test['month'] = test.date.str[5:7]

test['day'] = test.date.str[8:]

del test['date']





train = train.merge(people, right_on="people_id", left_on="people_id")

test = test.merge(people, right_on="people_id", left_on="people_id")





train['people_id'] = train['people_id'].str.extract('(\d+)', expand=True)

test['people_id'] = test['people_id'].str.extract('(\d+)', expand=True)





cols = train.columns[1:].tolist() + ['people_id']

train = train[cols]

test = test[cols]





ids = test['activity_id']

del train['activity_id']

del test['activity_id']



X_train = train.as_matrix()

X_test = test.as_matrix()

X_train





dtrain = xgb.DMatrix(X_train, y_train, missing=np.NaN)

dtest = xgb.DMatrix(X_test, missing=np.NaN)



params = {"objective": "binary:logistic", "eval_metric": "auc", "booster" : "gblinear",

          "eta": 0.05, "max_depth": 10, "subsample": 0.7, "colsample_bytree": 0.7}



num_boost_round = 100



gbm = xgb.train(params, dtrain, num_boost_round)

pred = gbm.predict(dtest)



print(gbm.eval(dtrain))



pred = [0 if p < 0.5 else 1 for p in pred]



submission = pd.DataFrame({'activity_id':ids, 'outcome': pred})

submission.to_csv('submission.csv', index=False)
