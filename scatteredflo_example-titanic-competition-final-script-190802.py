import numpy as np

import pandas as pd
train = pd.read_csv("../input/train.csv", index_col = ["PassengerId"])

print(train.shape)

train.head()
test = pd.read_csv("../input/test.csv", index_col = ["PassengerId"])

print(test.shape)

test.head()
submission = pd.read_csv("../input/gender_submission.csv")

print(submission.shape)

submission.head()
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
train['Sex']=train['Sex'].replace("female",1)

train['Sex']=train['Sex'].replace("male",0)



test['Sex']=test['Sex'].replace("female",1)

test['Sex']=test['Sex'].replace("male",0)

test.head()
train = train.drop("Name",1)

train = train.drop("Ticket",1)

train = train.drop("Cabin",1)



test = test.drop("Name",1)

test = test.drop("Ticket",1)

test = test.drop("Cabin",1)

test.head()
train['Embarked'].unique()
train_embarked = pd.get_dummies(train['Embarked'], prefix = 'embark')

train = pd.concat([train, train_embarked], axis=1)

train = train.drop("Embarked",1)



test_embarked = pd.get_dummies(test['Embarked'], prefix = 'embark')

test = pd.concat([test, test_embarked], axis=1)

test = test.drop("Embarked",1)

test.head()
train['Pclass'].unique()
train_Pclass = pd.get_dummies(train['Pclass'], prefix = 'Pclass')

train = pd.concat([train, train_Pclass], axis=1)

train = train.drop("Pclass",1)



train_Pclass = pd.get_dummies(test['Pclass'], prefix = 'Pclass')

test = pd.concat([test, train_Pclass], axis=1)

test = test.drop("Pclass",1)

test.head()
y = train['Survived']

X = train.drop(['Survived'],1)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=0)
import lightgbm as lgb

lgbm= lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

                         importance_type='split', learning_rate=0.01, max_depth=7                         ,

                         min_child_samples=10, min_child_weight=0.001, min_split_gain=0.0,

                         n_estimators=200, n_jobs=-1, num_leaves=30, objective='binary',

                         random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                         subsample=1.0, subsample_for_bin=200000, subsample_freq=0)





#=== DLP에서는 아래로 Model을 활용한다. ===#

# lgbm = lgb.LGBMRegressor(objective='regression', num_leaves=144, 

#                          learning_rate=0.005,n_estimators=720,max_depth=13,

#                          metric='rmse', is_training_metric=True, max_bin=55,

#                          bagging_fraction=0.8, verbose=-1, bagging_freq=5, feature_fraction=0.9)

lgbm.fit(X_train, y_train)
lgbm.fit(X_train, y_train)





pred_train = lgbm.predict(X_train)

print((pred_train == y_train).mean())

#=== DLP에서는 아래로 Score를 구해준다. ===#

# print(mean_squared_error(pred_train, y_train))





pred_valid= lgbm.predict(X_valid)

print((pred_valid == y_valid).mean())

#=== DLP에서는 아래로 Score를 구해준다. ===#

# print(mean_squared_error(pred_valid, y_valid))
pred_test=lgbm.predict(test)
submission.head()
submission = submission.drop("Survived",1)

##### 앞에서 이야기한 것 처럼 현재 Survived는 예시이므로 삭제해준다



pred_test = pd.DataFrame(pred_test)



submission_final = pd.concat([submission,pred_test],axis=1)

##### 우리가 예측한 값을 submission dataset에 합쳐준다 (axis=1 -> Column 합침)



submission_final.columns = ['PassengerId','Survived']

submission_final.to_csv("submission_final.csv", index=False)

submission_final.head()