#Import Libraries

import numpy as np

import pandas as pd

import lightgbm as lgbm

import xgboost as xb



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error
#Load the data

train = pd.read_csv("/kaggle/input/cais-exec-team-in-house/train.csv")

test = pd.read_csv("/kaggle/input/cais-exec-team-in-house/test.csv")

submission = pd.read_csv("/kaggle/input/cais-exec-team-in-house/sampleSubmission.csv")
#Lets divide the data into different column types

numeric_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',

                   'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

string_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',

                  'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',

                  'nursery', 'higher', 'internet', 'romantic', 'subject']

target_column = ['grade']
#We can't work with strings directly, so lets encode them into a form we can

encoder = LabelEncoder()



#Iterate through each column and encode as necessary

for string_column in string_columns:

    train[string_column] = encoder.fit_transform(train[string_column])

    test[string_column] = encoder.transform(test[string_column])
#Now lets split of the train into training (70&) and validation (30%)

training = train.sample(frac=0.7,random_state=69)

validation = train.drop(training.index)

training = train
#Since we are using lightgbm we need to create a specialized dataset

lgbm_train = lgbm.Dataset(training[numeric_columns + string_columns], training[target_column])
#Now I am going to specify my hyperparameters

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'l2',

    'num_leaves': 31,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}

# params = {

#     'boosting_type': 'gbdt',

#     'objective': 'regression',

#     'metric': 'l2',

# }
#Train the lgbm model

lgbm_model = lgbm.train(params, lgbm_train, num_boost_round=128)
xg_model = xb.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=1024,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 

xg_model = xb.XGBRegressor() 

xg_model.fit(training[numeric_columns + string_columns], training[target_column])
#Now lets prediction on the test data

grades = lgbm_model.predict(test[numeric_columns + string_columns], num_iteration=lgbm_model.best_iteration)

grades += xg_model.predict(test[numeric_columns + string_columns])

grades /= 2



#I pulled a sneaky and reformulated the problem as classification, so we need to get the output

submission["grade"] =grades # np.argmax(grades, axis=1)



#Save our submission and submi

submission.to_csv("submission.csv", index=False)