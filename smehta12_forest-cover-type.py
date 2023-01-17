# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import scipy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv(r"../input/train.csv")
test = pd.read_csv(r"../input/test.csv")
pd.set_option('display.width', 700)
train.head(5)
test.head(5)
print(train.dtypes)
train.describe()
print("The train data shape before any operation on the data: {} ".format(train.shape))
print("The test data shape before any operation on the data: {} ".format(test.shape))
# Drop Id col as it is not helpful in the classification

train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)
# Find out the frequency of nulls in the columns

# For training Data
count_nans = len(train) - train.count()
count_nans = count_nans.to_frame()
count_nans.columns=["train_nan_count"]
count_nans["%_train_nans"]=(count_nans["train_nan_count"]/train.shape[0]) * 100

# For test data
count_nans["test_nan_count"] = len(test) - test.count()
count_nans["%_test_nans"]=(count_nans["test_nan_count"]/test.shape[0]) * 100

count_nans.sort_values("train_nan_count", ascending=False, inplace=True)
count_nans.query('train_nan_count > 0 or test_nan_count > 0')
# Only for Training data
corr_matrix = train.corr().abs()
corr_matrix.dropna(axis=1, how="all", inplace=True) # remove the columns which is all Nan
corr_matrix.dropna(axis=0, how="all", inplace=True) # remove the rows which is all Nan
plt.subplots(figsize=(15,10))
sns.heatmap(corr_matrix, cmap="jet")
train.drop("Hillshade_3pm", inplace=True, axis=1)
test.drop("Hillshade_3pm", inplace=True, axis=1)
# take out training target data for further computation
y_train = train["Cover_Type"]
train.drop(["Cover_Type"], inplace=True, axis=1)
train.columns
# Find skewness in the y_train
y_train.hist()
# Checking Non-Binary columns only
skew_check_cols = ["Elevation", "Aspect", "Slope", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Hydrology", 
                   "Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Horizontal_Distance_To_Fire_Points" ]
train[skew_check_cols].hist(figsize=(25, 25))
# skewness = train[skew_check_cols].skew().abs().sort_values(ascending=False)
# print(skewness)
# skewed_cols = skewness[skewness>0.5].index.tolist()
# print(skewed_cols)
# train[skewed_cols] = train[skewed_cols].apply(np.log1p)
train[skew_check_cols].skew().abs().sort_values(ascending=False)
# ####################### Train data #############################################
# train['HF1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
# train['HF2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
# train['HR1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
# train['HR2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
# train['FR1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
# train['FR2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
# train['ele_vert'] = train.Elevation-train.Vertical_Distance_To_Hydrology

# train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
# train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

# #Mean distance to Amenities 
# train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3 
# #Mean Distance to Fire and Water 
# train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2 

# ####################### Test data #############################################
# test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
# test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
# test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
# test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
# test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
# test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])
# test['ele_vert'] = test.Elevation-test.Vertical_Distance_To_Hydrology

# test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
# test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

# #Mean distance to Amenities 
# test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3 
# #Mean Distance to Fire and Water 
# test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2


# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import RobustScaler

# train = RobustScaler(with_centering=True, with_scaling=True).fit_transform(train, y_train)
# test = RobustScaler(with_centering=True, with_scaling=True).fit_transform(test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


def kfold_cv(train, model, y_true):
    accuracy = []
    kfold = StratifiedKFold(n_splits=10, random_state=None)
    for tr, te in kfold.split(train, y_true):
        model.fit(train.iloc[tr], y_true.iloc[tr])
        predictions = model.predict(train.iloc[te])
        accuracy.append(accuracy_score(predictions, y_true.iloc[te]))
    
    print("KFold Accuracy:{}".format(np.mean(accuracy)))

def calculate_metrics(y_true, y_pred):
    #acc_score = accuracy_score(y_true, y_pred)
    print("accuracy score:{}".format(accuracy_score(y_true, y_pred)))
    print("f1 score:{}".format(f1_score(y_true, y_pred, average="micro")))
    # ROC Curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# TODO: Try multiscoring
def _grid_search(model, param_grid, train, y_train):
    clf = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', refit=True)
    clf.fit(train, y_train)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    print(clf.scorer_)
    print(clf.cv_results_)
from sklearn.linear_model import LogisticRegression
print("Logistic_Regression")
lr = LogisticRegression()
lr.fit(train, y_train)
y_train_pred = lr.predict(train)
y_test_pred = lr.predict(test)

kfold_cv(train, lr, y_train)
calculate_metrics(y_train, y_train_pred)
# from sklearn.ensemble import GradientBoostingClassifier
# print("Gradient Boost")
# cl = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
# cl.fit(train, y_train)
# y_train_pred = cl.predict(train)
# y_test_pred = cl.predict(test)

# # print("classifier score {}".format(cl.score(train, y_train)))

# kfold_cv(train, cl, y_train)
# calculate_metrics(y_train, y_train_pred)
# import sys
# orig_stdout=sys.stdout
# sys.stdout=open('output.txt', 'w')
import lightgbm as lgb

# d_train = lgb.Dataset(train, label=y_train)
# params = {}
# params['learning_rate'] = 0.003
# params['boosting_type'] = 'gbdt'
# params['objective'] = 'binary'
# params['metric'] = 'binary_logloss'
# params['sub_feature'] = 0.5
# params['num_leaves'] = 10
# params['min_data'] = 50
# params['max_depth'] = 10

# print("lgb_classifier")
# param_grid = {'n_estimators':[500, 600, 800], 'learning_rate':[0.1], 'max_depth':[40, 60], 'num_leaves':[127, 255], 'boosting_type':['gbdt']}
# cl = lgb.LGBMClassifier()
# _grid_search(cl, param_grid, train, y_train)

cl = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, max_depth=40, n_estimators=500, num_leaves=63, objective="multiclass")
cl.fit(train, y_train)
y_train_pred = cl.predict(train)
y_test_pred = cl.predict(test)

kfold_cv(train, cl, y_train)
calculate_metrics(y_train, y_train_pred)
from sklearn.metrics import classification_report
print(classification_report(y_train, y_train_pred))
from xgboost import XGBClassifier
# print("xgb_classifier")
# param_grid = {'n_estimators':[70, 300, 500], 'learning_rate':[0.001, 0.01, 0.1], 'max_depth':[2, 10, 20]}
# cl = XGBClassifier(n_jobs=10)
# _grid_search(cl, param_grid, train, y_train)

# cl = XGBClassifier(n_jobs=20, n_estimators=1000, learning_rate=0.1, max_depth=100)
# cl.fit(train, y_train)
# y_train_pred = cl.predict(train)
# y_test_pred = cl.predict(test)

# # print("classifier score {}".format(cl.score(train, y_train)))

# kfold_cv(train, cl, y_train)
# calculate_metrics(y_train, y_train_pred)
# from sklearn.ensemble import ExtraTreesClassifier

# cl = ExtraTreesClassifier(n_estimators=400)#, oob_score=True, bootstrap=True)
# cl.fit(train, y_train)
# y_train_pred = cl.predict(train)
# y_test_pred = cl.predict(test)

# #print("oob score {}".format(cl.oob_score_))

# kfold_cv(train, cl, y_train)
# calculate_metrics(y_train, y_train_pred)
idx = pd.read_csv("../input/test.csv").Id
my_submission = pd.DataFrame({'Id': idx, 'Cover_Type': y_test_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
my_submission.head()