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

#Extend the range of pandas colmns

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 80)
base_file_path = "/kaggle/input/house-prices-advanced-regression-techniques/"

train_df = pd.read_csv(base_file_path + "train.csv", index_col="Id")

test_df = pd.read_csv(base_file_path + "test.csv", index_col="Id")

print(train_df.shape)

train_df.head()
from feature_selector_kaggle import FeatureSelector

dependent_col = "SalePrice"

fs_start = FeatureSelector(data = train_df.drop(dependent_col, axis=1), labels = train_df[dependent_col])

fs_start.identify_missing(missing_threshold = 0.6)
fs_start.missing_stats.head()
fs_start.identify_single_unique()
#Remove columns found

single_missing_cols = fs_start.check_removal()

test_df.drop(single_missing_cols, axis=1, inplace=True)

train_df.drop(single_missing_cols, axis=1, inplace=True)
num_cols = list(train_df._get_numeric_data().columns)

cat_cols = list(set(train_df.columns) - set(num_cols))

len(num_cols) + len(cat_cols) == len(train_df.columns)
#Fill in missing for cat/num columns

#Simple fill for now; Imputing could be more useful in the future

cat_cols_dict = dict((col, 'Missing') for col in cat_cols)

num_cols_dict = dict((col, 0) for col in num_cols)

train_df.fillna(value=dict(**cat_cols_dict, **num_cols_dict), inplace=True)

test_df.fillna(value=dict(**cat_cols_dict, **num_cols_dict), inplace=True)
#Verify filled

nans_df = pd.DataFrame(train_df.isnull().sum(axis=0), columns=["na_count"])

nans_df["pct_missing"] = nans_df["na_count"] / len(train_df)

nans_df[nans_df["na_count"] > 0].sort_values(by="na_count", ascending=False)
#Combine categoricals that are observed below the ratio as "other"

cat_replace_dict = {}

def get_categorical_combine(col,ratio):

    obsv_count = len(col)

    obsv_ratio = col.value_counts() /  obsv_count

    obsv_ratio_fail = obsv_ratio[obsv_ratio < ratio]

    repalce_str = "|".join(obsv_ratio_fail.index)

    cat_replace_dict[col.name] = repalce_str

    return 



train_df[cat_cols].apply(get_categorical_combine,ratio=.05,axis=0)

train_df_trim = train_df.replace(cat_replace_dict,"other", regex=True)

test_df_trim = test_df.replace(cat_replace_dict,"other", regex=True)
#Encode ordinal and categorical variables

import category_encoders as ce

def categorical_features(train_df: pd.DataFrame, test_df:pd.DataFrame, dependent_col: str, to_skip: list, num_cats: list, bayes:list):

    #Drop skip list

    train_df = train_df.drop(to_skip, axis=1)

    test_df = test_df.drop(to_skip, axis=1)

    

    ##Save Dep Col

    dep_col_series = train_df[dependent_col].copy()

    

    #Drop Dep Col for Encoding

    train_df = train_df.drop(dependent_col, axis=1)



    #Get list of categorical columns, only based on type

    categorical = list(train_df.select_dtypes("object").columns)

    

    #Add in those that are numerical categoricals (i.e, nieghborhood identifier)

    categorical = categorical + num_cats



    #Remove those used for Baysien Encoding

    categorical = list(set(categorical) - set(bayes))

    

    #Check for harighly cardinal and use binary encoding

    to_binary = []

    for var in categorical:

        if train_df[var].nunique() > 10:

            to_binary.append(var)

            print("Use Binary Encoder on ", var)



    ##Binary

    # instantiate an encoder - here we use Binary()

    ce_binary = ce.BinaryEncoder(cols=to_binary)

    ce_binary.fit(train_df)





    #Use binary encoder

    train_df = ce_binary.transform(train_df)

    test_df = ce_binary.transform(test_df)



    #Baysien

    for var in bayes:

        print("Use Baysien Encoder on ", var)



    #Use fit bayes encoder

    ce_james = ce.JamesSteinEncoder(cols=bayes)

    ce_james.fit(train_df, dep_col_series)



    # Use bayes encoder

    train_df = ce_james.transform(train_df)

    test_df = ce_james.transform(test_df)



    #One Hot

    one_hot_cols = set(categorical) - set(to_binary)

    train_df = pd.get_dummies(train_df, columns=one_hot_cols)

    test_df = pd.get_dummies(test_df, columns=one_hot_cols)



    #Add dependent column back

    train_df[dependent_col] = dep_col_series

    

    return train_df, test_df



train_df_encoded, test_df_encoded = categorical_features(train_df_trim, test_df_trim, "SalePrice", [], [], ["GarageCond", "OverallQual","OverallCond"])

train_df_encoded.head()
#Remove any columns not found in both dataframes after encoding

unused_features_train = list(set(train_df_encoded.columns) - set(test_df_encoded.columns))

unused_features_train.remove("SalePrice")

train_df_encoded.drop(unused_features_train, axis=1,inplace=True)



unused_features_test = list(set(test_df_encoded.columns) - set(train_df_encoded.columns))

test_df_encoded.drop(unused_features_test, axis=1,inplace=True)
fs = FeatureSelector(data = train_df_encoded.drop(dependent_col, axis=1), labels = train_df_encoded[dependent_col])

fs.identify_collinear(correlation_threshold = 0.98)
# Pass in the appropriate parameters

fs.identify_zero_importance(task = 'regression', 

                            eval_metric = 'auc', 

                            n_iterations = 10, 

                             early_stopping = True)
#Collect columns to drop

colinear_noimportance_cols = fs.check_removal()

train_df_encoded.drop(colinear_noimportance_cols, axis=1, inplace=True)

test_df_encoded.drop(colinear_noimportance_cols, axis=1, inplace=True)
from sklearn.ensemble import ExtraTreesClassifier

dependent_col = "SalePrice"

X = train_df_encoded.drop(dependent_col, axis=1)

y = train_df_encoded[dependent_col]

clf = ExtraTreesClassifier(n_estimators=100, random_state=0)

clf.fit(X,y)

y_hat = clf.predict(test_df_encoded)
submit_df = pd.concat([pd.Series(test_df_encoded.index), pd.Series(y_hat)], axis=1)

submit_df.columns = ["Id", "SalePrice"]

submit_df.head()
%%capture

import pandas as pd

import numpy as np

from sklearn.model_selection import (StratifiedKFold, KFold,

                                     ParameterGrid)

import xgboost

from sklearn.metrics import mean_squared_error



CLASS = False  # Whether classification or regression

SCORE_MIN = True  # Optimizing score through minimum

k = 5  # Number of folds

best_score = 10

best_params = None

best_iter = None



# CV

train_feats = train_df_encoded.drop(dependent_col, axis=1)

target = train_df_encoded[dependent_col]

test_feats = test_df_encoded



train = np.array(train_feats)

target = np.log(np.array(target))  # Changes to Log

test = np.array(test_feats)

print(train.shape, test.shape)



if CLASS:

    kfold = StratifiedKFold(target, k)

else:

    kfold_base = KFold()

    kfold = list(kfold_base.split(train))



early_stopping = 50



param_grid = [

              {'verbosity': [0],

               'nthread': [2],

               'eval_metric': ['rmse'],

               'eta': [0.03],

               'objective': ['reg:linear'],

               'max_depth': [5, 7],

               'num_round': [1000],

               'subsample': [0.2, 0.4, 0.6],

               'colsample_bytree': [0.3, 0.5, 0.7],

               }

              ]



# Hyperparmeter grid optimization

param_num = 0

for params in ParameterGrid(param_grid):

    print(params)

    # Determine best n_rounds

    xgboost_rounds = []

    for train_index, test_index in kfold:

        X_train, X_test = train[train_index], train[test_index]

        y_train, y_test = target[train_index], target[test_index]



        xg_train = xgboost.DMatrix(X_train, label=y_train)

        xg_test = xgboost.DMatrix(X_test, label=y_test)



        watchlist = [(xg_train, 'train'), (xg_test, 'test')]



        num_round = params['num_round']

        xgclassifier = xgboost.train(params, xg_train, num_round,

                                     watchlist,

                                     early_stopping_rounds=early_stopping);

        xgboost_rounds.append(xgclassifier.best_iteration)



    if len(xgboost_rounds) == 0:  

        num_round = 0

    else:

        num_round = int(np.mean(xgboost_rounds))

    

    # Solve CV

    rmsle_score = []

    for cv_train_index, cv_test_index in kfold:

        X_train, X_test = train[cv_train_index, :], train[cv_test_index, :]

        y_train, y_test = target[cv_train_index], target[cv_test_index]



        # train machine learning

        xg_train = xgboost.DMatrix(X_train, label=y_train)

        xg_test = xgboost.DMatrix(X_test, label=y_test)



        watchlist = [(xg_train, 'train'), (xg_test, 'test')]



        xgclassifier = xgboost.train(params, xg_train, num_round);



        # predict

        predicted_results = xgclassifier.predict(xg_test)

        rmsle_score.append(np.sqrt(mean_squared_error(y_test, predicted_results)))



    if SCORE_MIN:

        if best_score > np.mean(rmsle_score):

            print(np.mean(rmsle_score))

            print('new best')

            best_score = np.mean(rmsle_score)

            best_params = params

            best_iter = num_round

    else:

        if best_score < np.mean(rmsle_score):

            print(np.mean(rmsle_score))

            print('new best')

            best_score = np.mean(rmsle_score)

            best_params = params

            best_iter = num_round

    

    #Iterate param index 

    param_num+=1



# Solution using best parameters

print('best params: %s' % best_params)

print('best score: %f' % best_score)

xg_train = xgboost.DMatrix(train, label=target)

xg_test = xgboost.DMatrix(test)

watchlist = [(xg_train, 'train')]

num_round = best_iter  # already int

xgclassifier = xgboost.train(best_params, xg_train, num_round, watchlist);
#Prepare XGBoost Submission

submission_name = '../input/house-prices-advanced-regression-techniques/sample_submission.csv'

submission_col = 'SalePrice'

submission_target = 'xgboost.csv'

submission = pd.read_csv(submission_name)

submission[submission_col] = np.exp(xgclassifier.predict(xg_test))

submission.to_csv(submission_target, index=False)

submission.head()
from tpot import TPOTRegressor

X_train, X_test, y_train, y_test = train_test_split(train_df_encoded.drop(dependent_col, axis=1),

                                                    train_df_encoded[dependent_col],

                                                    train_size=0.75, test_size=0.25, random_state=42)



tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)

tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))

tpot.export('tpot_boston_pipeline.py')
submission[submission_col] = tpot.predict(test_feats)

submission.head()
submission[submission_col] = np.exp(xgclassifier.predict(xg_test)) *.5 + tpot.predict(test_feats)*.5

submission.head()
submission.to_csv("submit_mixed.csv", index=False)