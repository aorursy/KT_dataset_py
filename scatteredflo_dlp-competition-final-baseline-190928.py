

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pd.set_option('display.max_columns', 10000)

train = pd.read_csv("/kaggle/input/train.csv")

train = train.drop(['ID'],1)     # train의 ID를 Drop 시켜줌

train = train.reset_index(drop=True)     # train의 index를 초기화 시켜줌

print(train.shape)

train.head()
test = pd.read_csv("/kaggle/input/test.csv")

test = test.drop(['ID'],1)     # train의 ID를 Drop 시켜줌

test = test.reset_index(drop=True)     # train의 index를 초기화 시켜줌

print(test.shape)

test.head()
submission = pd.read_csv("/kaggle/input/submission.csv")

print(submission.shape)

submission.head()
list(train.dtypes[train.dtypes == 'object'].index)
# from sklearn.preprocessing import LabelEncoder

# cols = ['A','B']



# for c in cols:

#     lbl = LabelEncoder() 

#     lbl.fit(list(train[c].values)) 

#     train[c] = lbl.transform(list(train[c].values))     # unique_column을 전무 label_encoding함



# print('Shape all_data: {}'.format(train.shape))

# from sklearn.preprocessing import LabelEncoder

# cols = ['A','B']



# for c in cols:

#     lbl = LabelEncoder() 

#     lbl.fit(list(test[c].values)) 

#     test[c] = lbl.transform(list(test[c].values))     # unique_column을 전무 label_encoding함



# print('Shape all_data: {}'.format(test.shape))

train = train.drop(['A','B'],1)

test = test.drop(['A','B'],1)
train.head()
# train['END_TM'] = pd.to_datetime(train['END_TM'], format='%Y-%m-%d %H:%M:%S')

# train['year'] = train['END_TM'].dt.year

# train['month'] = train['END_TM'].dt.month

# train['day'] = train['END_TM'].dt.day

# train['weekday'] = train['END_TM'].dt.weekday # (0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일)

# # datetime을 정상적으로 만드는 방법

# train.head()
# test['END_TM'] = pd.to_datetime(test['END_TM'], format='%Y-%m-%d %H:%M:%S')

# test['year'] = test['END_TM'].dt.year

# test['month'] = test['END_TM'].dt.month

# test['day'] = test['END_TM'].dt.day

# test['weekday'] = test['END_TM'].dt.weekday # (0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일)

# # datetime을 정상적으로 만드는 방법

# test.head()
train = train.drop(['END_TM'],1)

test = test.drop(['END_TM'],1)
y = pd.DataFrame(train['Y'])

y.head()
X = train.drop(['Y'],1)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split (X,y, random_state=0)
print(X.shape, y.shape, "-->", X_train.shape, y_train.shape, X_valid.shape,y_valid.shape)
import lightgbm as lgb

lgbm = lgb.LGBMRegressor (objective = 'regression', num_leaves=144,

                         learning_rate=0.005,n_estimators=720, max_depth=13,

                         metric='rmse', is_training_metric=True, max_bin=55,

                         bagging_fraction=0.8, verbose=-1, bagging_freq=5, feature_fraction=0.9)
lgbm.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error



# #=== MSE ===#

# pred_train = lgbm.predict(X_train)

# pred_valid = lgbm.predict(X_valid)

# print(mean_squared_error(pred_train, y_train))

# print(mean_squared_error(pred_valid, y_valid))



#=== RMSE ===#



pred_train = lgbm.predict(X_train)

pred_valid = lgbm.predict(X_valid)



def rmse(predictions, targets):

    return np.sqrt(mean_squared_error(predictions, targets))



print(rmse(pred_train, y_train))

print(rmse(pred_valid, y_valid))

# 위에는 MSE이며, RMSE를 구할때는 해당으로 진행
(pred_train - y_train['Y']).mean()
pd.DataFrame(pred_train)
pd.DataFrame(y_train)
test.head()
pred_test = lgbm.predict(test)

pred_test
submission = submission.drop("Y",1)

pred_test = pd.DataFrame(pred_test)



submission_final = pd.concat([submission,pred_test],axis=1)



submission_final.columns = ['ID','Y']

submission_final.to_csv("submission_fianl.csv", index=False)

submission_final.tail()