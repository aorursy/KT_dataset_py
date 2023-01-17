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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
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
                                               ('num',num_transformer,num_cols),
                                               ('cat',cat_transformer,cat_cols)
])
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD

model_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
t_SVD = TruncatedSVD(n_components=5,n_iter=7,random_state=42)
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('t_SVD',t_SVD),
                              ('model_RF', model_RF)])
rf_pipeline.fit(X_train,y_train)
scores_RF = cross_val_score(rf_pipeline,X,y,cv=10,scoring='roc_auc')
score_RF = scores_RF.mean()
print(score_RF)
