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
print(X_train_r.shape)
print(X_test_r.shape)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
folds=StratifiedKFold(n_splits=2,random_state=1)
test_pred=np.empty((X_test_r.shape[0],1),float)
train_pred=np.empty((0,1),float)
print(test_pred.shape)
print(train_pred.shape)
print(X_train_r.shape)
print(X_test_r.shape)
for train_indices,val_indices in folds.split(X_train_r,y_train.values):
    x_train,x_val=X_train_r.iloc[train_indices],X_train_r.iloc[val_indices]
    y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]
    print(x_train,x_val,y_train,y_val)
    model.fit(X=x_train,y=y_train)
    x_train_pred = model.predict(x_val)
    print(x_train_pred)
    train_pred=np.append(train_pred,x_train_pred)
    x_test_pred = model.predict(test)
    print(x_test_pred)
    test_pred=np.append(test_pred,x_test_pred)
    
    

def Stacking(model,train,y,test,n_fold):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((test.shape[0],1),float)
    train_pred=np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

        model.fit(X=x_train,y=y_train)
        train_pred=np.append(train_pred,model.predict(x_val))
        test_pred=np.append(test_pred,model.predict(test))
    return test_pred.reshape(-1,1),train_pred
model_DTC = DecisionTreeClassifier(random_state=1)

test_pred_DTC ,train_pred_DTC = Stacking(model=model_DTC,n_fold=10, train=X_train_r,test=X_test_r,y=y_train)

train_pred_DTC=pd.DataFrame(train_pred_DTC)
test_pred_DTC=pd.DataFrame(test_pred_DTC)
print(X_train_r.shape)
print(train_pred_DTC.shape)
print(test_pred_DTC.shape)
model_KNN = KNeighborsClassifier()

test_pred_KNN ,train_pred_KNN =Stacking(model=model_KNN,n_fold=10,train=X_train_r,test=X_test_r,y=y_train)

train_pred_KNN=pd.DataFrame(train_pred_KNN)
test_pred_KNN=pd.DataFrame(test_pred_KNN)
X_train_cmb = pd.concat([train_pred_DTC, train_pred_KNN], axis=1)
X_test_cmb = pd.concat([test_pred_DTC, test_pred_KNN], axis=1)

model_LR = LogisticRegression(random_state=1)
model_LR.fit(X_train_cmb,y_train)
model_LR.score(X_test_cmb, y_test)
print(X_train_cmb.shape)
print(y_train.shape)
print(X_test_cmb.shape)
print(y_test.shape)
y_pred = model_LR.predict(X_test_cmb)
score = roc_auc_score(y_test,y_pred)
print(score)
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
