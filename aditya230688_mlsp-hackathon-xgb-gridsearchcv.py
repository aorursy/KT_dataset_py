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
agent_df = pd.read_csv('/kaggle/input/mlsp-hackathon/train_HK6lq50.csv')
y = agent_df.is_pass
X = agent_df.drop(['is_pass','id','test_id'],axis=1)
cat_cols = [cname for cname in X.columns if X[cname].dtypes == 'object' and X[cname].nunique() < 25]
num_cols = [cname for cname in X.columns if X[cname].dtypes in ['int64','float64']]
print (cat_cols)
print (num_cols)
#Utilize only these columns for Model configuration
use_cols = cat_cols + num_cols
X = X[use_cols].copy()

#Preprocessing Numerical columns - Missing values 
def find_null_col(dataset):
    null_col = dataset.columns[dataset.isnull().any()]
    null_col_sum = dataset[null_col].isnull().sum()
    return null_col,null_col_sum
  
#impute by groupby on following columns -
group_col1 = ['trainee_id']
group_col2 = ['gender','education','city_tier']

#UDF to impute Null values with the mean of the group of meaningful columns.
def impute_null(dataset,null_col,group_col):
    for col in null_col:
        dataset[col] = dataset[col].fillna(dataset.groupby(group_col)[col].transform('mean'))

null_col_X,null_col_X_sum = find_null_col(X)
impute_null(X,null_col_X,group_col1)
impute_null(X,null_col_X,group_col2)
null_col_X,null_col_X_sum = find_null_col(X)
print("Null Columns :" ,null_col_X)
print("Null Columns info :",null_col_X_sum)
X.drop(['trainee_id'],axis=1,inplace=True)
num_col_x = ['program_duration', 'city_tier', 'age', 'total_programs_enrolled', 'trainee_engagement_rating']
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
my_num_imputer = SimpleImputer(strategy='mean')
SIM_X_train = pd.DataFrame(my_num_imputer.fit_transform(X_train[num_col_x]))
SIM_X_test = pd.DataFrame(my_num_imputer.transform(X_test[num_col_x]))
SIM_X_train.columns = X_train[num_col_x].columns
SIM_X_test.columns = X_test[num_col_x].columns
#SIM_X_train
my_num_scaler = StandardScaler()
SSM_X_train = pd.DataFrame(my_num_scaler.fit_transform(SIM_X_train))
SSM_X_test = pd.DataFrame(my_num_scaler.transform(SIM_X_test))
SSM_X_train.columns = SIM_X_train.columns
SSM_X_test.columns = SIM_X_test.columns
#SSM_X_train
my_cat_imputer = SimpleImputer(strategy='most_frequent')
SIC_X_train = pd.DataFrame(my_cat_imputer.fit_transform(X_train[cat_cols]))
SIC_X_test = pd.DataFrame(my_cat_imputer.transform(X_test[cat_cols]))
SIC_X_train.columns = X_train[cat_cols].columns
SIC_X_test.columns = X_test[cat_cols].columns
#SIC_X_train
my_ohe_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
OHE_X_train = pd.DataFrame(my_ohe_encoder.fit_transform(SIC_X_train))
OHE_X_test = pd.DataFrame(my_ohe_encoder.transform(SIC_X_test))
#OHE removes index, so we put it back
OHE_X_train.index = SIC_X_train.index
OHE_X_test.index = SIC_X_test.index
#OHE_X_train
X_train_r = pd.concat([SSM_X_train,OHE_X_train],axis=1)
X_test_r = pd.concat([SSM_X_test,OHE_X_test],axis=1)
import xgboost as xgb
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate, GridSearchCV

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
def modelfit(alg, X_train,y_train,useTrainCV=True, cv_folds=5, early_stopping_rounds=10):
        
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(y_train.values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
#Choose all predictors except target & IDcols
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=200,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

modelfit(xgb1, X_train_r, y_train)
y_pred1 = xgb1.predict(X_test_r)
score1 = roc_auc_score(y_test,y_pred1)
print(score1)
print(xgb1)
#Use n_estimators = 140, since it produced best result
#Tune max_depth and min_child_weight
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)
gsearch1.fit(X_train_r,y_train)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
#Tune Gamma
param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4, cv=5)
gsearch2.fit(X_train_r,y_train)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
#Re-calibrate
xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=140,
 max_depth=9,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb2, X_train_r, y_train)
y_pred2 = xgb2.predict(X_test_r)
score2 = roc_auc_score(y_test,y_pred2)
print(score2)
print(xgb2)
# Tune subsample and colsample_bytree
param_test3 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=1, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4, cv=5)
gsearch3.fit(X_train_r,y_train)
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
#Tuning Regularisation parameters
param_test4 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0.0, subsample=0.8, colsample_bytree=0.9,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4, cv=5)
gsearch4.fit(X_train_r,y_train)
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_
#Re-calibrate
xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=140,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.9,
 reg_alpha=0.01,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, X_train_r, y_train)
y_pred3 = xgb3.predict(X_test_r)
score3 = roc_auc_score(y_test,y_pred3)
print(score3)
print(xgb3)
#Reducing the learning rate
xgb4 = XGBClassifier(
 learning_rate =0.05,
 n_estimators=140,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, X_train_r, y_train)
y_pred4 = xgb4.predict(X_test_r)
score4 = roc_auc_score(y_test,y_pred4)
print(score4)
print(xgb4)
from sklearn.externals import joblib
joblib.dump(xgb4, 'MLSP_Hackathon_XGB_GCV.pkl')
test_df = pd.read_csv('/kaggle/input/mlsp-hackathon/test_wF0Ps6O.csv')
test_df.drp(['test_id'],axis=1,inplace=True)
#Removing the ID column and storing in another frame
X_test_f = test_df.iloc[:,1:15]
X_test_idf = test_df.iloc[:,0:1]
print(X_test_f.shape)
print(X_test_idf.shape)
cat_cols = [cname for cname in X_test_f.columns if X_test_f[cname].dtypes == 'object' and X_test_f[cname].nunique() < 25]
num_cols = [cname for cname in X_test_f.columns if X_test_f[cname].dtypes in ['int64','float64']]
print (cat_cols)
print (num_cols)
#Utilize only these columns for Model configuration
use_cols = cat_cols + num_cols
X_test_f = X_test_f[use_cols].copy()

null_col_X,null_col_X_sum = find_null_col(X_test_f)
impute_null(X_test_f,null_col_X,group_col1)
impute_null(X_test_f,null_col_X,group_col2)
null_col_X,null_col_X_sum = find_null_col(X_test_f)
print("Null Columns :" ,null_col_X)
print("Null Columns info :",null_col_X_sum)

X_test_f.drop(['trainee_id'],axis=1,inplace=True)

my_num_imputer = SimpleImputer(strategy='mean')
SIM_X_test_f = pd.DataFrame(my_num_imputer.fit_transform(X_test_f[num_col_x]))
SIM_X_test_f.columns = X_test_f[num_col_x].columns

my_num_scaler = StandardScaler()
SSM_X_test_f = pd.DataFrame(my_num_scaler.fit_transform(SIM_X_test_f))
SSM_X_test_f.columns = SIM_X_test_f.columns

my_cat_imputer = SimpleImputer(strategy='most_frequent')
SIC_X_test_f = pd.DataFrame(my_cat_imputer.fit_transform(X_test_f[cat_cols]))
SIC_X_test_f.columns = X_test_f[cat_cols].columns

my_ohe_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
OHE_X_test_f = pd.DataFrame(my_ohe_encoder.fit_transform(SIC_X_test_f))
OHE_X_test_f.index = SIC_X_test_f.index

X_test_fr = pd.concat([SSM_X_test_f,OHE_X_test_f],axis=1)
MLSP_model_XGB_GCV = joblib.load('MLSP_Hackathon_XGB_GCV.pkl')
y_test_fr = MLSP_model_XGB_GCV.predict(X_test_fr)
submission_df = pd.read_csv('/kaggle/input/mlsp-hackathon/sample_submission_vaSxamm.csv')
submission_df['is_pass'] = y_test_fr
submission_df.to_csv('MSLP_Hackathon_XGB_GCV_sub.csv',index=False)
