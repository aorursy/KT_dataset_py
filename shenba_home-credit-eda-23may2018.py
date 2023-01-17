# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/application_train.csv')
train_df.shape
#there are 307K rows and 122 columns; the number of cols is very high
train_df.head()
# we see that TARGET is the output variable and also that we have a number of demographic variables
#let's explore the output variable
train_df.TARGET.value_counts()
# low event rate (as we would expect)
print("event rate is : {} %".format(round((train_df.TARGET.value_counts()[1]/train_df.shape[0]) * 100)))
#the file homecredit_columns_description has details 
train_df.NAME_CONTRACT_TYPE.value_counts()
#there are a number of numeric and categorical features; for the baseline model in this notebook, let's take only a few
#also, for the categorical features, we will try to do mean encoding instead of the regular one hot encoding / label encoding
num_features_list = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED']
cat_features_list = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                     'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'OCCUPATION_TYPE']
#mean encoding for cat features
num_rows = train_df.shape[0] #get number of records in train

for cat_feature in cat_features_list: #iterate over all the cat features
        encoder_series = train_df[cat_feature].value_counts() / num_rows #create a series that would have the mean for each value in the cat feature
        train_df[cat_feature+'_mean_enc'] = train_df[cat_feature].map(encoder_series) #map that to the specific cat feature and create a new col
# we have created a set of cols that are mean encoded from categorical cols
#lets move on to create a baseline model with the selected numeric features and the mean encoded cols
#create a list with numeric cols
train_df.columns
features = num_features_list + ['NAME_CONTRACT_TYPE_mean_enc', 'CODE_GENDER_mean_enc', 'FLAG_OWN_CAR_mean_enc', 'FLAG_OWN_REALTY_mean_enc',
                               'NAME_INCOME_TYPE_mean_enc', 'NAME_EDUCATION_TYPE_mean_enc',
                               'NAME_FAMILY_STATUS_mean_enc', 'OCCUPATION_TYPE_mean_enc']
X_train = train_df[features]
y_train = train_df.TARGET
from xgboost import XGBClassifier
seed = 111
#without scale pos weight, we had no 1 preds; with scale pos weight as 12, we had a 127K 1s with accuracy of 60%
model_xgb = XGBClassifier(scale_pos_weight=6)
model_xgb.fit(X=X_train, y=y_train)
np.sum(model_xgb.predict(X_train))
from sklearn.metrics import accuracy_score
accuracy_score(y_true=y_train, y_pred=model_xgb.predict(data=X_train))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true=y_train, y_pred=model_xgb.predict(data=X_train))
#now to make predictions on the test set
#first, we need to do the mean encoding for the test data as well
test_df = pd.read_csv('../input/application_test.csv')
for cat_feature in cat_features_list: #iterate over all the cat features
        test_df[cat_feature+'_mean_enc'] = test_df[cat_feature].map(encoder_series) #map that to the specific cat feature and create a new col
X_test = test_df[features]
y_pred_test = model_xgb.predict(X_test)
np.sum(y_pred_test)
#no 1s are predicted; FAIL
#do submission
y_pred_test_prob = model_xgb.predict_proba(X_test)[:, 1]


Submission = pd.DataFrame({ 'SK_ID_CURR': test_df.SK_ID_CURR,'TARGET': y_pred_test_prob })
Submission.to_csv("sample_submission_baseline_23May18.csv", index=False)
