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



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np



# Datavisulizing 

import matplotlib.pyplot as plt

import seaborn as sns

# import seaborn as sn

import missingno as msno





from sklearn.impute import SimpleImputer

import scipy

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import auc, roc_curve

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from tqdm import tqdm_notebook





from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import category_encoders as ce



from sklearn import metrics



import lightgbm as lgb

raw_train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv', index_col='id')

raw_test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv', index_col='id')

raw_submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv', index_col='id')

print(raw_train.shape, raw_test.shape, raw_submission.shape)
raw_train.head()
raw_train.columns
#Rename columns for traningset 

structuret_train = raw_train.copy();



structuret_train = structuret_train.rename(columns = {'bin_0':'zerosundones_0'});

structuret_train = structuret_train.rename(columns = {'bin_1':'zerosundones_1'});

structuret_train = structuret_train.rename(columns = {'bin_2':'zerosundones_2'});

structuret_train = structuret_train.rename(columns = {'bin_3':'FvsT'});

structuret_train = structuret_train.rename(columns = {'bin_4':'NvsY'});



structuret_train = structuret_train.rename(columns = {'nom_0': 'colors'});

structuret_train = structuret_train.rename(columns = {'nom_1': 'trigonometry'});

structuret_train = structuret_train.rename(columns = {'nom_2': 'animals'});

structuret_train = structuret_train.rename(columns = {'nom_3': 'contries'});

structuret_train = structuret_train.rename(columns = {'nom_4': 'instruments'});

structuret_train = structuret_train.rename(columns = {'nom_5': 'random_0'});

structuret_train = structuret_train.rename(columns = {'nom_6': 'random_1'});

structuret_train = structuret_train.rename(columns = {'nom_7': 'random_2'});

structuret_train = structuret_train.rename(columns = {'nom_8': 'random_3'});

structuret_train = structuret_train.rename(columns = {'nom_9': 'random_4'});



structuret_train = structuret_train.rename(columns = {'ord_0': 'oneTwoTree'});

structuret_train = structuret_train.rename(columns = {'ord_1': 'competetitions_levels'});

structuret_train = structuret_train.rename(columns = {'ord_2': 'temperature'});

structuret_train = structuret_train.rename(columns = {'ord_3': 'alpha_0'});

structuret_train = structuret_train.rename(columns = {'ord_4': 'alpha_1'});

structuret_train = structuret_train.rename(columns = {'ord_5': 'alpha_2'});
#Rename columns for testset 

structuret_test = raw_test.copy();



structuret_test = structuret_test.rename(columns = {'bin_0':'zerosundones_0'});

structuret_test = structuret_test.rename(columns = {'bin_1':'zerosundones_1'});

structuret_test = structuret_test.rename(columns = {'bin_2':'zerosundones_2'});

structuret_test = structuret_test.rename(columns = {'bin_3':'FvsT'});

structuret_test = structuret_test.rename(columns = {'bin_4':'NvsY'});



structuret_test = structuret_test.rename(columns = {'nom_0': 'colors'});

structuret_test = structuret_test.rename(columns = {'nom_1': 'trigonometry'});

structuret_test = structuret_test.rename(columns = {'nom_2': 'animals'});

structuret_test = structuret_test.rename(columns = {'nom_3': 'contries'});

structuret_test = structuret_test.rename(columns = {'nom_4': 'instruments'});

structuret_test = structuret_test.rename(columns = {'nom_5': 'random_0'});

structuret_test = structuret_test.rename(columns = {'nom_6': 'random_1'});

structuret_test = structuret_test.rename(columns = {'nom_7': 'random_2'});

structuret_test = structuret_test.rename(columns = {'nom_8': 'random_3'});

structuret_test = structuret_test.rename(columns = {'nom_9': 'random_4'});



structuret_test = structuret_test.rename(columns = {'ord_0': 'oneTwoTree'});

structuret_test = structuret_test.rename(columns = {'ord_1': 'competetitions_levels'});

structuret_test = structuret_test.rename(columns = {'ord_2': 'temperature'});

structuret_test = structuret_test.rename(columns = {'ord_3': 'alpha_0'});

structuret_test = structuret_test.rename(columns = {'ord_4': 'alpha_1'});

structuret_test = structuret_test.rename(columns = {'ord_5': 'alpha_2'});
msno.matrix(raw_train, figsize = (30,5))
investigate_structuret_train = pd.DataFrame({'columns':structuret_train.columns})

investigate_structuret_train['data_type'] = structuret_train.dtypes.values

investigate_structuret_train['missing_val'] = structuret_train.isnull().sum().values 

investigate_structuret_train['uniques'] = structuret_train.nunique().values

investigate_structuret_train

target_dist = structuret_train.target.value_counts()



barplot = plt.bar(target_dist.index, target_dist, color = 'darkred', alpha = 0.8)

barplot[0].set_color('darkblue')



plt.xlabel('Target', fontsize = 18)



plt.show()

print("percentage of the target: {}%".format(structuret_train.target.sum() / len(structuret_train.target)))
plt.figure(figsize=(12,10))

cols = raw_train.select_dtypes(exclude=['object']).columns

data = raw_train[cols].corr()

sns.heatmap(data, 

            xticklabels=data.columns.values,

            yticklabels=data.columns.values)



plt.show()
# Diving the data into train and validation and selecting ther target value 

def feature_target(DataFrame):

    

    DataFrame.fillna(DataFrame.median(), inplace = True)

    

    y = DataFrame.target 

    X = DataFrame.drop(['target'], axis = 1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=0)

    

    return X_train, X_valid, y_train, y_valid 
def test_method(X_train, X_valid, y_train, y_valid):

    

    train = lgb.Dataset(X_train, label=y_train)

    valid = lgb.Dataset(X_valid, label=y_valid)

    

    param = {'num_leaves': 64, 'objective': 'binary', 

             'metric': 'auc', 'seed': 7}

    

    model_bst = lgb.train(param, train, num_boost_round=1000, valid_sets=[valid], 

                    early_stopping_rounds=10, verbose_eval=False)



    valid_pred = model_bst.predict(X_valid)

    

    mae = mean_absolute_error(y_valid, valid_pred)

    

    print(f"Validation MAE score: {mae:.4f}")

    

    return model_bst
#creating a list with all the categorical data with relatively low cardinality

categorical_data = [cat for cat in structuret_train if

                    structuret_train[cat].nunique() < 10 and 

                    structuret_train[cat].dtype == "object"]



categorical_data_test = [cat for cat in structuret_test if

                    structuret_test[cat].nunique() < 10 and 

                    structuret_test[cat].dtype == "object"]



# We want to find the numerical data

numerical_data = [num for num in structuret_train if structuret_train[num].dtype in ['int64', 'float64']]

numerical_data_test = [num for num in structuret_test if structuret_test[num].dtype in ['int64', 'float64']]
# making a copy of the renamed columns from raw_train(the original trainings data)

label_train = structuret_train.copy() 

onehot_train = structuret_train.copy()

count_train = structuret_train.copy()

target_train = structuret_train.copy()

catboost_train = structuret_train.copy()



# making a copy of the renamed columns from raw_test(the original test data)

label_test = structuret_test.copy() 

onehot_test = structuret_test.copy()

count_test = structuret_test.copy()

target_test = structuret_test.copy()

catboost_test = structuret_test.copy()
# converting all enteries to strings

label_train[categorical_data] = label_train[categorical_data].astype(str)  

label_test[categorical_data] = label_test[categorical_data].astype(str) 

    

# Label encoding

cat_features = categorical_data

encoder = LabelEncoder()



label_encoded_train = label_train[cat_features].apply(encoder.fit_transform)

label_encoded_test = label_test[cat_features].apply(encoder.fit_transform)



label_train = label_train[numerical_data].join(label_encoded_train)

label_test = label_test[numerical_data_test].join(label_encoded_test)

X_t, X_v, y_t, y_v = feature_target(label_train)

bst_results_label = test_method(X_t, X_v, y_t, y_v)
count_train[categorical_data] = count_train[categorical_data].astype(str)

count_train[numerical_data] = count_train[numerical_data].astype(float)

count_test[categorical_data_test] = count_test[categorical_data_test].astype(str)

count_test[numerical_data_test] = count_test[numerical_data_test].astype(float)
# Count encoding 

cat_features = categorical_data



count_enc = ce.CountEncoder()



count_encoded_train = count_enc.fit_transform(count_train[cat_features])

count_encoded_test = count_enc.fit_transform(count_test[cat_features])



count_encoded_train = label_train.join(count_encoded_train.add_suffix("_count"))

count_encoded_test = label_test.join(count_encoded_test.add_suffix("_count"))
X_t, X_v, y_t, y_v = feature_target(count_encoded_train)

bst_results_count = test_method(X_t, X_v, y_t, y_v)
cat_features = categorical_data



target_enc = ce.TargetEncoder(cols=cat_features)



target_encoded_train = target_enc.fit_transform(target_train[cat_features], target_train.target)

target_encoded_test = target_enc.transform(target_test[cat_features])



target_encoded_train = count_encoded_train.join(target_encoded_train.add_suffix("_target"))

target_encoded_test = count_encoded_test.join(target_encoded_test.add_suffix("_target"))
X_t, X_v, y_t, y_v = feature_target(target_encoded_train)

bst_results_target = test_method(X_t, X_v, y_t, y_v)
cat_features = categorical_data



catboost_enc = ce.CatBoostEncoder(cols=cat_features)



catboost_encoded_train = catboost_enc.fit_transform(catboost_train[cat_features], catboost_train.target)

catboost_encoded_test = catboost_enc.transform(catboost_test[cat_features])



catboost_encoded_train = count_encoded_train.join(catboost_encoded_train.add_suffix("_catboost"))

catboost_encoded_test = count_encoded_test.join(catboost_encoded_test.add_suffix("_catboost"))
X_t, X_v, y_t, y_v = feature_target(catboost_encoded_train)

bst_results_catboost = test_method(X_t, X_v, y_t, y_v)
from sklearn.feature_selection import SelectKBest, f_classif





feature_cols = target_encoded_train.columns.drop('target')

X_t, X_v, y_t, y_v = feature_target(target_encoded_train)



# Keeping 14 features

selector = SelectKBest(f_classif, k=13)



X_new = selector.fit_transform(X_t, y_t)

X_new
# Get back the features we want to kept

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=X_t.index, 

                                 columns=feature_cols)

selected_features.head()
# Dropping columns that has zero values

selected_columns = selected_features.columns[selected_features.var() != 0]



# using valid for the selected features.

X_v[selected_columns].head()
# Testing the effect of dropping the feature that has no influence on the target value 

investigate = pd.DataFrame(target_encoded_train[selected_columns])

investigate['target'] = target_encoded_train.target



investigate_test = pd.DataFrame(target_encoded_test[selected_columns])



X_t, X_v, y_t, y_v = feature_target(investigate)

bst = test_method(X_t, X_v, y_t, y_v)
# optimizing model RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression



RFR_model = RandomForestRegressor(n_estimators = 100, random_state = 0)



X_train_1, X_valid_1, y_train_1, y_valid_1 = feature_target(investigate)



RFR_model.fit(X_train_1, y_train_1)  

preds_1 = RFR_model.predict(X_valid_1) 

RFR_mae = mean_absolute_error(preds_1, y_valid_1) 

RFR_mae
from sklearn.linear_model import LogisticRegression



X_train_2, X_valid_2, y_train_2, y_valid_2 = feature_target(investigate)

                                                    

LR_model = LogisticRegression(C=0.03, max_iter=300)

LR_model.fit(X_train_2, y_train_2)

preds_2 = LR_model.predict_proba(X_valid_2)[:, 1]



LR_mae = mean_absolute_error(preds_2, y_valid_2) 
LR_mae
import xgboost as xgb



X_train_3, X_valid_3, y_train_3, y_valid_3 = feature_target(investigate)



xgb_model = xgb.XGBClassifier(max_depth=20,n_estimators=2020,colsample_bytree=0.20,learning_rate=0.020,objective='binary:logistic', n_jobs=-1)



xgb_model.fit(X_train_3, y_train_3,eval_set=[(X_valid_3,y_valid_3)],verbose=0,early_stopping_rounds=200

) 



# preds_3 = xgb_model.predict(X_valid_3) 

preds_3 = xgb_model.predict_proba(X_valid_3)[:,1]

xgb_mae = mean_absolute_error(preds_3, y_valid_3) 

xgb_mae
import lightgbm as lgb



X_train_4, X_valid_4, y_train_4, y_valid_4 = feature_target(investigate)



train = lgb.Dataset(X_train_4, label=y_train_4)

valid = lgb.Dataset(X_valid_4, label=y_valid_4)

    

param = {'num_leaves': 64, 'objective': 'binary', 

             'metric': 'auc', 'seed': 7}

    

lgb_model = lgb.train(param, train, num_boost_round=1000, valid_sets=[valid], 

                    early_stopping_rounds=10, verbose_eval=False)



preds_4 = lgb_model.predict(X_valid_4)

    

lgb_mae = mean_absolute_error(y_valid_4, preds_4)
lgb_mae
scores = [RFR_mae, LR_mae, xgb_mae, lgb_mae]

pd.DataFrame(np.array([scores]),

                   columns=['RFR_mae', 'LR_mae', 'xgb_mae', 'lgb_mae'])

test_pred_0 = lgb_model.predict(investigate_test)
submission_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')

sub_id = submission_df['id']

submission = pd.DataFrame({'id':sub_id})

submission['target'] = test_pred_0

submission.to_csv("submission5.csv",index = False)

print('Model ready for submission!')