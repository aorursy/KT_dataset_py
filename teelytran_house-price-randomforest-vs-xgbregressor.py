# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
X_train_full=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
X_test_full=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
X_train_full.shape
X_train_full.head()
X_train_full.info()

X_test_full.shape
X_test_full.head()
X_test_full.info()

from matplotlib import pyplot as plt
import seaborn as sns
cor_mat=X_train_full.corr()
sns.set(rc={'figure.figsize':(20,12)})
sns.heatmap(cor_mat,linewidths=.5,cmap="YlGnBu")
plt.show()
high_cor_cols=list(cor_mat['SalePrice'][cor_mat['SalePrice'].values<1].nlargest(10).index)
high_cor_cols
X_train_full.drop('Id', axis=1, inplace=True)
test_Id=X_test_full['Id']
X_test_full.drop('Id', axis=1, inplace=True)

X_train_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y= X_train_full.SalePrice
X_train_full.drop('SalePrice', axis=1, inplace=True)
train_missing_counts=X_train_full.isnull().count()
train_missing_sum=X_train_full.isnull().sum()
percent=train_missing_sum/train_missing_counts
train_missing_df=pd.DataFrame({'Sum':train_missing_sum,'Percentage': percent}).sort_values(by='Percentage', ascending=False)
train_missing_df.head(20)
to_drop=['PoolQC','MiscFeature','Alley','Fence']
X_train_full.drop(to_drop,axis=1, inplace=True)
X_test_full.drop(to_drop, axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid=train_test_split(X_train_full, y, random_state=0, test_size=0.2)

object_cols=list(X_train.select_dtypes(include='object').columns)
X_train[object_cols].nunique().sort_values(ascending=False)
cat_cols=[col for col in X_train.columns if X_train[col].nunique() < 10 and X_train[col].dtype=='object']
num_cols=[col for col in X_train.columns if X_train[col].dtype != 'object']
to_keep=cat_cols+num_cols

X_train=X_train[to_keep]
X_valid=X_valid[to_keep]
X_test=X_test_full[to_keep]
X_train.shape
X_train.columns
X_valid.shape
X_valid.columns
X_test.shape
X_test.columns
X_train[cat_cols].isnull().sum().sort_values(ascending=False)
replace_with_NA=['FireplaceQu','GarageCond','GarageQual','GarageFinish','GarageType','BsmtFinType2','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1']
for col in replace_with_NA:
    X_train[col]=X_train[col].fillna('NA')
    X_valid[col]=X_valid[col].fillna('NA')
from scipy.stats import norm
sns.set(rc={'figure.figsize':(10,5)})
sns.distplot(y, fit=norm)
y_train=np.log(y_train)
y_valid=np.log(y_valid)
X_train['House_age']=pd.Series(X_train['YrSold']-X_train['YearBuilt'])
X_valid['House_age']=pd.Series(X_valid['YrSold']-X_valid['YearBuilt'])
X_test['House_age']=pd.Series(X_test['YrSold']-X_test['YearBuilt'])
X_train.House_age.describe()
X_train.House_age.isnull().sum()

num_cols.append('House_age')
num_cols
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

cat_transformer=Pipeline(steps=[('Imputer',SimpleImputer(strategy='most_frequent')),
                                ('Encoder',OneHotEncoder(handle_unknown='ignore'))])
num_transformer=Pipeline(steps=[('Imputer',SimpleImputer(strategy='median')),('Scaler',StandardScaler())])
preprocessor=ColumnTransformer(transformers=[('cat',cat_transformer,cat_cols),
                                             ('num',num_transformer,num_cols)])
rf_model=RandomForestRegressor(n_estimators=100, random_state=0)
xgb_model=XGBRegressor(random_state=0,n_estimators=100, learning_rate=0.05)
def get_score(model):
    pipeline=Pipeline(steps=[('processor',preprocessor),('model',model)])
    pipeline.fit(X_train, y_train)
    y_pred=pipeline.predict(X_valid)
    score=mean_absolute_error(y_valid, y_pred)
    return score
print('RandomForestRegressor Score: {}'.format(get_score(rf_model)))
print('XGBRegressor Score: {}'.format(get_score(xgb_model)))
from sklearn.model_selection import GridSearchCV
rf_param_grids={'n_estimators':np.arange(100,400,50)}
xgb_param_grids={'n_estimators':np.arange(100,400,50),'max_depth':np.arange(3,8), 'learning_rate':[0.01,0.02,0.05]}
rf_model=RandomForestRegressor(random_state=0)
xgb_model=XGBRegressor(random_state=0)
rf_model_cv=GridSearchCV(rf_model, rf_param_grids, cv=5)
xgb_model_cv=GridSearchCV(xgb_model, xgb_param_grids, cv=5)
print('RandomForestRegressor Score: {}'.format(get_score(rf_model_cv)))
print('RandomForestRegressor Best Parameters: {}'.format(rf_model_cv.best_params_))

print('XGBRegressor Score: {}'.format(get_score(xgb_model_cv)))
print('XGBRegressor Best Parameters: {}'.format(xgb_model_cv.best_params_))
my_pipeline=Pipeline(steps=[('processor',preprocessor),('model',xgb_model_cv)])
predictions=my_pipeline.predict(X_test)
predictions=np.expm1(predictions)
submission=pd.DataFrame({'Id':test_Id,'SalePrice':predictions})
submission.head()
submission.to_csv('submission.csv', index=False)