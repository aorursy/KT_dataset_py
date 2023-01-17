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
X = agent_df.drop(['is_pass'],axis=1)
cat_cols = [cname for cname in X.columns if X[cname].dtypes == 'object' and X[cname].nunique() < 25]
num_cols = [cname for cname in X.columns if X[cname].dtypes in ['int64','float64']]
print (cat_cols)
print (num_cols)
#Utilize only these columns for Model configuration
use_cols = cat_cols + num_cols
X = X[use_cols].copy()
X = X[['program_id','program_type','program_duration','test_id','test_type','difficulty_level','trainee_id','gender','education','city_tier','age','total_programs_enrolled','is_handicapped','trainee_engagement_rating']]

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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
my_num_imputer = SimpleImputer(strategy='mean')
SIM_X_train = pd.DataFrame(my_num_imputer.fit_transform(X_train[num_cols]))
SIM_X_test = pd.DataFrame(my_num_imputer.transform(X_test[num_cols]))
SIM_X_train.columns = X_train[num_cols].columns
SIM_X_test.columns = X_test[num_cols].columns
SIM_X_train
my_num_scaler = StandardScaler()
SSM_X_train = pd.DataFrame(my_num_scaler.fit_transform(SIM_X_train))
SSM_X_test = pd.DataFrame(my_num_scaler.transform(SIM_X_test))
SSM_X_train.columns = SIM_X_train.columns
SSM_X_test.columns = SIM_X_test.columns
SSM_X_train
my_cat_imputer = SimpleImputer(strategy='most_frequent')
SIC_X_train = pd.DataFrame(my_cat_imputer.fit_transform(X_train[cat_cols]))
SIC_X_test = pd.DataFrame(my_cat_imputer.transform(X_test[cat_cols]))
SIC_X_train.columns = X_train[cat_cols].columns
SIC_X_test.columns = X_test[cat_cols].columns
SIC_X_train
my_ohe_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
OHE_X_train = pd.DataFrame(my_ohe_encoder.fit_transform(SIC_X_train))
OHE_X_test = pd.DataFrame(my_ohe_encoder.transform(SIC_X_test))
#OHE removes index, so we put it back
OHE_X_train.index = SIC_X_train.index
OHE_X_test.index = SIC_X_test.index
OHE_X_train
X_train_r = pd.concat([SSM_X_train,OHE_X_train],axis=1)
X_test_r = pd.concat([SSM_X_test,OHE_X_test],axis=1)
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
#Bayesian optimization
def bayesian_optimization(dataset, function, parameters):
   X_train_r, y_train, X_test_r, y_test = dataset
   n_iterations = 5
   gp_params = {"alpha": 1e-4}

   BO = BayesianOptimization(function, parameters)
   BO.maximize(n_iter=n_iterations, **gp_params)

   return BO.max
def rfc_optimization(cv_splits):
    def function(n_estimators, max_depth, min_samples_split):
        return cross_val_score(
               RandomForestClassifier(
                   n_estimators=int(max(n_estimators,0)),                                                               
                   max_depth=int(max(max_depth,1)),
                   min_samples_split=int(max(min_samples_split,2)), 
                   n_jobs=-1, 
                   random_state=42,   
                   class_weight="balanced"),  
                   X=X_train_r, 
                   y=y_train, 
                   cv=cv_splits,
                   scoring="roc_auc",
    ).mean()

    parameters = {"n_estimators": (10, 1000),
                  "max_depth": (1, 150),
                  "min_samples_split": (2, 10)}
    
    return function, parameters
def model_RF(dataset, function, parameters):
        best_solution = bayesian_optimization(dataset, function, parameters)      
        params = best_solution["params"]
        classifier_RF = RandomForestClassifier(
                 n_estimators=int(max(params["n_estimators"], 0)),
                 max_depth=int(max(params["max_depth"], 1)),
                 min_samples_split=int(max(params["min_samples_split"], 2)), 
                 n_jobs=-1, 
                 random_state=42,   
                 class_weight="balanced")
        return classifier_RF
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train_r, y_train)
dataset = (X_train_res, y_train_res, X_test_r, y_test)
cv_splits = 10

function, parameters = rfc_optimization(cv_splits)
rf_model = model_RF(dataset, function, parameters)
rf_model.fit(X_train_res,y_train_res)
y_pred = rf_model.predict(X_test_r)
score = roc_auc_score(y_test,y_pred)
print(score)
from sklearn.externals import joblib
joblib.dump(rf_model, 'MLSP_Hackathon_RFClassifier_BO.pkl')
test_df = pd.read_csv('/kaggle/input/mlsp-hackathon/test_wF0Ps6O.csv')
test_df.head()
#Removing the ID column from the test dataset
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
X_test_f = X_test_f[['program_id','program_type','program_duration','test_id','test_type','difficulty_level','trainee_id','gender','education','city_tier','age','total_programs_enrolled','is_handicapped','trainee_engagement_rating']]

null_col_X,null_col_X_sum = find_null_col(X)
impute_null(X,null_col_X,group_col1)
impute_null(X,null_col_X,group_col2)
null_col_X,null_col_X_sum = find_null_col(X)
print("Null Columns :" ,null_col_X)
print("Null Columns info :",null_col_X_sum)

my_num_imputer = SimpleImputer(strategy='mean')
SIM_X_test_f = pd.DataFrame(my_num_imputer.fit_transform(X_test_f[num_cols]))
SIM_X_test_f.columns = X_test_f[num_cols].columns

my_num_scaler = StandardScaler()
SSM_X_test_f = pd.DataFrame(my_num_scaler.fit_transform(SIM_X_test_f))
SSM_X_test_f.columns = SIM_X_test_f.columns

my_cat_imputer = SimpleImputer(strategy='most_frequent')
SIC_X_test_f = pd.DataFrame(my_cat_imputer.fit_transform(X_test_f[cat_cols]))
SIC_X_test_f.columns = X_test_f[cat_cols].columns

my_ohe_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
OHE_X_test_f = pd.DataFrame(my_ohe_encoder.fit_transform(SIC_X_test_f))
#OHE removes index, so we put it back
OHE_X_test_f.index = SIC_X_test_f.index

X_test_fr = pd.concat([SSM_X_test_f,OHE_X_test_f],axis=1)

MLSP_model_RFBO = joblib.load('MLSP_Hackathon_RFClassifier_BO.pkl')
y_test_fr = MLSP_model_RFBO.predict(X_test_fr)
submission_df = pd.read_csv('/kaggle/input/mlsp-hackathon/sample_submission_vaSxamm.csv')
submission_df['is_pass'] = y_test_fr
submission_df.to_csv('MSLP_Hackathon_RFClassifier_sub3.csv',index=False)
