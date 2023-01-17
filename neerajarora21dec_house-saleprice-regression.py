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
train_csv = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_csv = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib inline
train_csv.head()
test_csv.head()
train_csv.shape
test_csv.shape
df = pd.concat((train_csv,test_csv))
df.shape
df.tail()
df.head()
df.info()
df = df.set_index("Id")
df.head()
plt.figure(figsize=(20,10))
sns.heatmap(df.isnull())
df.shape
null_list = (df.isnull().sum()/2919)*100
null_list[null_list >10].keys()
null_list[null_list >10].size
df.columns.size
df = df.drop(null_list[null_list >10].keys(),"columns")

train_csv.corr()
plt.figure(figsize=(20,10))
sns.heatmap(train_csv.corr())
corr = train_csv.corr()
corr
high_corr = corr.index[abs(corr['SalePrice']) > 0.5]
high_corr
plt.figure(figsize=(20,10))
sns.heatmap(train_csv[high_corr].corr())

sns.pairplot(train_csv[high_corr])
missing_cols = df[df.columns[df.isnull().any()]]
missing_cols.columns
Basement_features =['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1',
       'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']


missing_cols[Basement_features].info()
missing_cols[Basement_features].isnull().sum()
missing_cols[Basement_features]
Basement_cat_features = ['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']
Basement_num_features = ['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF']
missing_cols[Basement_num_features] = missing_cols[Basement_num_features].fillna(0)
missing_cols[Basement_cat_features] = missing_cols[Basement_cat_features].fillna('NA')
missing_cols[Basement_features].isnull().sum()
df[Basement_features] = missing_cols[Basement_features]
missing_cols.head()
Garage_features = ['GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt']
missing_cols['GarageCars'].fillna(1.0,inplace =True)
missing_cols['GarageArea'].fillna(300,inplace =True)
df['GarageCars'] = missing_cols['GarageCars']
df['GarageArea'] = missing_cols['GarageArea']
missing_cols[Garage_features].isnull().sum()
garage_feat = missing_cols[Garage_features]
garage_feat_all_nan = garage_feat[(garage_feat.isnull() | garage_feat.isin([0])).all(1)]
garage_feat_all_nan.shape
for i in garage_feat:
    if i in Garage_features:
        garage_feat_all_nan[i] = garage_feat_all_nan[i].replace(np.nan, 'NA')
    else:
        garage_feat_all_nan[i] = garage_feat_all_nan[i].replace(np.nan, 0)
        
garage_feat.update(garage_feat_all_nan)
missing_cols.update(garage_feat_all_nan)
garage_feat = garage_feat[garage_feat.isnull().any(axis=1)]
garage_feat
for i in Garage_features:
    garage_feat[i] = garage_feat[i].replace(np.nan, df[df['GarageType'] == 'Detchd'][i].mode()[0])
garage_feat.isnull().any()
missing_cols.update(garage_feat)
df[Garage_features] = missing_cols[Garage_features]
missing_cols.head(50)
missing_cols.isnull().sum()
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
df['Exterior1st'].fillna(df['Exterior1st'].mode()[0], inplace=True)
df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0], inplace=True)
df['KitchenQual'].fillna(df['KitchenQual'].mode()[0], inplace=True)
df['MSZoning'].fillna(df['MSZoning'].mode()[0], inplace=True)
df['SaleType'].fillna(df['SaleType'].mode()[0], inplace=True)
df['Utilities'].fillna(df['Utilities'].mode()[0], inplace=True)
df['Functional'].fillna(df['Functional'].mode()[0], inplace=True)
missing_cols.head(50)
mas_feat_all_nan = missing_cols[(missing_cols.isnull()).all(1)]
mas_feat_all_nan.shape
mas_feat_all_nan['MasVnrArea'].fillna(0.0,inplace=True)
mas_feat_all_nan['MasVnrType'].fillna('None',inplace=True)
mas_feat_all_nan.shape
missing_cols[(missing_cols.isnull()).all(1)]
missing_cols.update(mas_feat_all_nan)

missing_cols['MasVnrType'].fillna('Stone',inplace=True)
df[['MasVnrType','MasVnrArea']] = missing_cols[['MasVnrType','MasVnrArea']]
df.shape
df[df['GarageYrBlt'] == 'NA']['GarageYrBlt'].count()
df['GarageYrBlt'] = df['GarageYrBlt'].replace({"NA": 0.0})
df.isnull().sum()
df.shape
df.columns[df.isnull().any()]
train_csv = train_csv.set_index("Id")
df2 = pd.concat([df, train_csv['SalePrice']], axis=1)
df2.shape
df2[df.dtypes[df.dtypes == np.object].index].columns
df2[df2.dtypes[df2.dtypes == np.object].index].columns
from pandas.api.types import CategoricalDtype

Qual_cond_cat_feat = ['BsmtCond','BsmtQual','GarageQual','GarageCond']
for i in Qual_cond_cat_feat:
    df2[i] = df[i].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df2[['Exterior1st', 'Exterior2nd']].corrwith(df2['SalePrice'])
df2['BsmtFinType1'] = df['BsmtFinType1'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ','GLQ'], ordered = True)).cat.codes
df2['BsmtFinType2'] = df['BsmtFinType2'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ','GLQ'], ordered = True)).cat.codes
df2['GarageFinish'] = df2['GarageFinish'].astype(CategoricalDtype(categories=['NA', 'Unf', 'RFn', 'Fin'], ordered = True)).cat.codes
df2['GarageType'] = df2['GarageType'].astype(CategoricalDtype(categories=['NA','Detchd', 'CarPort', 'BuiltIn', 'Basment','Attchd','2Types'], ordered = True)).cat.codes
df2['ExterQual'] = df2['ExterQual'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df2['ExterCond'] = df2['ExterCond'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df2.shape
df3 = df2.copy()
def category_onehot_multcols(multcolumns):
    df_final=df3
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(df3[fields],drop_first=True)
        
        df3.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([df3,df_final],axis=1)
        
    return df_final
df3[df3.dtypes[df3.dtypes == np.object].index].columns
final_df=category_onehot_multcols(df3[df3.dtypes[df3.dtypes == np.object].index].columns)
final_df.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]
train_len = len(train_csv)
X_train = final_df.drop(['SalePrice'],axis=1)[:train_len]
X_test = final_df.drop(['SalePrice'],axis=1)[train_len:]
y_train = final_df['SalePrice'][:train_len]

print(X_train.shape)
print(X_test.shape)
print(len(y_train))
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score
import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 1000, random_state=51)
test_model(rf_reg)
import xgboost
#xgb_reg=xgboost.XGBRegressor()
xgb_reg = xgboost.XGBRegressor(bbooster='gbtree', random_state=51)
test_model(xgb_reg)
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
base_score=[0.25,0.5,0.75,1]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
xgb_reg = xgboost.XGBRegressor()
random_cv = RandomizedSearchCV(estimator=xgb_reg,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
xgb_reg_tuned = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=0, max_depth=2,
             min_child_weight=1,monotone_constraints=None,
             n_estimators=900, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)

test_model(xgb_reg_tuned)
xgb_reg_tuned.fit(X_train,y_train)
y_pred = xgb_reg_tuned.predict(X_test)
y_pred
solution = pd.concat([test_csv['Id'], pd.DataFrame(y_pred)], axis=1)
solution.columns=['Id', 'SalePrice']
solution.head()
solution.to_csv('HousePrice_Regression_submission.csv', index=False)
