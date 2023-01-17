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

        

pd.set_option('display.max_columns', 10000)



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/dlp-private-competition-dataset-modificated/train.csv", index_col =["ID"])

test = pd.read_csv("/kaggle/input/dlp-private-competition-dataset-modificated/test.csv", index_col =["ID"])

submission = pd.read_csv("/kaggle/input/dlp-private-competition-dataset-modificated/submission.csv")





train.head()
test.head()
print(train.shape)

print(test.shape)
print(train.isnull().sum())

print(test.isnull().sum())
train = train.drop("END_TM",1)

test = test.drop("END_TM",1)

train.head()
print(train['A'].unique())

print(train['B'].unique())

print(test['A'].unique())

print(test['B'].unique())
train_A = pd.get_dummies(train['A'], prefix='A')

train = pd.concat([train, train_A], axis=1)

train = train.drop("A",1)

train_B = pd.get_dummies(train['B'], prefix='B')

train = pd.concat([train, train_B], axis=1)

train = train.drop("B",1)



test_A = pd.get_dummies(test['A'], prefix='A')

test = pd.concat([test, test_A], axis=1)

test = test.drop("A",1)

test_B = pd.get_dummies(test['B'], prefix='B')

test = pd.concat([test, test_B], axis=1)

test = test.drop("B",1)



train.head()
test.head()
print(train.shape)

print(test.shape)
test = test.drop("B_K",1)

# I romoved 1 column from test data because number of train_column & test_clolmn 

# must be same to run LGBM
print(train.shape)

print(test.shape)
# # [Feature Importance Finding Process - It takes 20min to run]

# save_to_file = pd.DataFrame(0., columns = ['MSE_train','MSE_valid','diff_train','diff_valid','diff_sum' ],

#                            index = ['C'+str(x) for x in range(1,118)])

# import matplotlib.pyplot as plt



# for each in range(1,118):

#     train_tmp = train.drop('C'+str(each),1)

    

#     y = train_tmp['Y']

#     X = train_tmp.drop(['Y'],1)

    

#     from sklearn.model_selection import train_test_split

#     X_train, X_valid, y_train, y_valid = train_test_split (X,y, random_state=0)

    

#     import lightgbm as lgb

#     lgbm = lgb.LGBMRegressor (objective = 'regression', num_leaves=144,

#                              learning_rate=0.005,n_estimators=720, max_depth=13,

#                              metric='rmse', is_training_metric=True, max_bin=55,

#                              bagging_fraction=0.8, verbose=-1, bagging_freq=5, feature_fraction=0.9)



#     lgbm.fit(X_train, y_train)

#     from sklearn.metrics import mean_squared_error

    

#     print(str(each))

#     pred_train = lgbm.predict(X_train)

#     train_mse = mean_squared_error(pred_train, y_train)

#     print(train_mse)

    

#     pred_valid = lgbm.predict(X_valid)

#     valid_mse = mean_squared_error(pred_valid, y_valid)

#     print(valid_mse)

    

#     ['MSE_train', 'MSE_valid', 'diff_train', 'diff_valid', 'diff_sum'] 

#     save_to_file['MSE_train']['C'+str(each)] = train_mse

#     save_to_file['MSE_valid']['C'+str(each)] = valid_mse  

#     save_to_file['diff_train']['C' + str(each)] = 8.77497 - train_mse

#     save_to_file['diff_valid']['C' + str(each)] = 18.6718 - valid_mse

#     save_to_file['diff_sum']['C' + str(each)] = (8.77497 - train_mse)+(18.6718 - valid_mse)

    

# save_to_file.to_csv("result_feature_importance.csv", index=False)



# plt.plot(save_to_file)

# plt.show()

# plt.savefig('result.jpg')

    
train = train.drop(["C59","C78","C93","C101"],1)

test = test.drop(["C59","C78","C93","C101"],1)

print(train.shape)

print(test.shape)
for x in range(1,11250):

    # set range 11250 among 13746 rows (not all)

    # x (row number) starts form 0 so +1

    try:

        if train['Y'][x+1]==0:

            train = train.drop(x+1)

    except:

        pass

# test set ID = 14,19,23, ... So vacant ID in train should pass in for function

# 223 rows that Y=0 are removed

train.shape
y = train['Y']

X = train.drop(['Y'],1)



import lightgbm as lgb

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split (X,y, random_state=0)



lgbm = lgb.LGBMRegressor (objective = 'regression', num_leaves=144,

                             learning_rate=0.0096,n_estimators=720, max_depth=13,

                             metric='rmse', is_training_metric=True, max_bin=55,

                             bagging_fraction=0.8, verbose=-1, bagging_freq=5, feature_fraction=0.9)

# I tried to optimize each factors but only learning_rate value made MSE value better (0.005 --> 0.0096)

# When I changed num leaves from 144 to 200, valid_mse decreased but final score increased

lgbm.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error





pred_train = lgbm.predict(X_train)

train_mse = mean_squared_error(pred_train, y_train)

print(train_mse)



pred_valid = lgbm.predict(X_valid)

valid_mse = mean_squared_error(pred_valid, y_valid)

print(valid_mse)
pred_test = lgbm.predict(test)

submission.head()
submission = submission.drop("Y",1)

pred_test = pd.DataFrame(pred_test)



submission_final = pd.concat([submission,pred_test],axis=1)



submission_final.columns = ['ID','Y']

submission_final.to_csv("submission_fianl.csv", index=False)

submission_final.head()