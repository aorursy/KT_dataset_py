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
#load the test set
test_df =  pd.read_csv('../input/application_test.csv')
#create flag in train and test df to identify them
train_df['is_train'] = 1
test_df['is_train'] = 0
#take the train output variable out from the train df so that we can merge train and test for processing
Y_train = train_df['TARGET']
train_X = train_df.drop(['TARGET'], axis = 1)
# test ID
test_id = test_df['SK_ID_CURR']
test_X = test_df

# merge train and test datasets for preprocessing
data = pd.concat([train_X, test_X], axis=0)
#write functions to get the categorical features in the overall dataset
# function to obtain Categorical Features
def get_categorical_features(df):
    cat_feats = [col for col in list(df.columns) if df[col].dtype == 'object']
    return cat_feats
#function to encode categorical values; we use pd.get_dummies;
#refer to https://www.kaggle.com/shivamb/homecreditrisk-extensive-eda-baseline-0-772; this nb has used both factorize and get dummies while I feel that just
#get dummies should do
def get_dummies(df, cat_feats):
    for cat_col in cat_feats:
        df = pd.concat([df, pd.get_dummies(df[cat_col], prefix=cat_col)], axis=1)
    return df
# get categorical features
data_cat_feats = get_categorical_features(data)
# create additional dummy features - 
data = get_dummies(data,data_cat_feats)
data.head()
#get numeric cols
numeric_cols = [col_name for col_name in list(data.columns) if data[col_name].dtype != 'object']
len(numeric_cols)
'is_train' in numeric_cols
numeric_cols = [col for col in numeric_cols if col !='is_train']
len(numeric_cols)
# remove the ID from list
numeric_cols = [col for col in numeric_cols if col !='SK_ID_CURR']
#split the data back in to train and test
train_X = data[data['is_train'] == 1][numeric_cols]
test_X = data[data['is_train'] == 0][numeric_cols]
from sklearn.model_selection import train_test_split 
random_seed = 144
#create validation sets to be used while training the model
X_train, X_val, y_train, y_val = train_test_split(train_X, Y_train, test_size=0.2, random_state=random_seed)
#build a simple light gbm model
import lightgbm as lgb
#prepare the train and eval data to fit to model
lgb_train = lgb.Dataset(data=X_train, label=y_train)
lgb_eval = lgb.Dataset(data=X_val, label=y_val)
#define the params for the model
params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          'colsample_bytree':.8, 'subsample':.9, 'max_depth':7, 'reg_alpha':.1, 'reg_lambda':.1, 
          'min_split_gain':.01, 'min_child_weight':1}
#used same params as here: https://www.kaggle.com/shivamb/homecreditrisk-extensive-eda-baseline-0-772
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)
#preds
preds = model.predict(test_X)
sub_lgb = pd.DataFrame()
sub_lgb['SK_ID_CURR'] = test_id
sub_lgb['TARGET'] = preds
sub_lgb.to_csv("lgb_baseline.csv", index=False)
sub_lgb.head()
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
