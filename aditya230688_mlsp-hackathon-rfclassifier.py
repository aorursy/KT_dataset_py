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
agent_df.drop(['id','test_id'],axis=1,inplace=True)
y = agent_df.is_pass
X = agent_df.drop(['is_pass'],axis=1)
X['program_id'].nunique()
cat_cols = [cname for cname in X.columns if X[cname].dtypes == 'object' and X[cname].nunique() < 25]
num_cols = [cname for cname in X.columns if X[cname].dtypes in ['int64','float64']]
print (cat_cols)
print (num_cols)
#Utilize only these columns for Model configuration
use_cols = cat_cols + num_cols
X = X[use_cols].copy()
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
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
print("Null Columns :" ,"\n",null_col_X)
print("Null Columns info :","\n",null_col_X_sum)
impute_null(X,null_col_X,group_col1)
impute_null(X,null_col_X,group_col2)
null_col_X,null_col_X_sum = find_null_col(X)
print("Null Columns :" ,null_col_X)
print("Null Columns info :",null_col_X_sum)
X = X.drop(['trainee_id'],axis=1)
num_col_x = ['program_duration', 'city_tier', 'age', 'total_programs_enrolled', 'trainee_engagement_rating']
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
#Preprocessing Numerical Data
num_transformer = Pipeline(steps=[
                                  ('imputer',SimpleImputer(strategy='mean')),
                                  ('scaler',StandardScaler())
                                  ])
#Preprocessing Categorical Data
cat_transformer = Pipeline(steps=[
                                  ('imputer',SimpleImputer(strategy='most_frequent')),
                                  ('onehot',OneHotEncoder(handle_unknown='ignore'))
                                  ])
#Bundle Preprocessing steps
preprocessor = ColumnTransformer(transformers=[
                                               ('num',num_transformer,num_col_x),
                                               ('cat',cat_transformer,cat_cols)
])
model_RF_1 = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced',
                       criterion='gini', max_depth=124, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=9,
                       min_weight_fraction_leaf=0.0, n_estimators=643,
                       n_jobs=-1, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)

#Pipeline for RFClassifier
pipeline_RF_1 = Pipeline(steps=[
                           ('preprocessor',preprocessor),
                           ('model_RF_1',model_RF_1)
                           
])
pipeline_RF_1.fit(X,y)
scores_RF_1 = cross_val_score(pipeline_RF_1,X,y,cv=10,scoring='roc_auc')
score_RF_1 = scores_RF_1.mean()
print(score_RF_1)
from sklearn.externals import joblib
joblib.dump(preprocessor, 'MLSP_Hackathon_Preprocessor.pkl')
joblib.dump(model_RF_1, 'MLSP_Hackathon_RFC_1.pkl')
test_df = pd.read_csv('/kaggle/input/mlsp-hackathon/test_wF0Ps6O.csv')
test_df.head()
#Removing the ID column from the test dataset
X_test = test_df.iloc[:,1:15]
X_test_id = test_df.iloc[:,0:1]
print(X_test.shape)
print(X_test_id.shape)
MLSP_Preprocessor = joblib.load('MLSP_Hackathon_Preprocessor.pkl')
X_test = MLSP_Preprocessor.transform(X_test)
MLSP_model_1 = joblib.load('MLSP_Hackathon_RFC_1.pkl')
y_test_1 = MLSP_model_1.predict(X_test)
submission_df_1 = pd.read_csv('/kaggle/input/mlsp-hackathon/sample_submission_vaSxamm.csv')
submission_df_1['is_pass'] = y_test_1
submission_df_1.to_csv('MSLP_Hackathon_RFC_1_sub.csv',index=False)
