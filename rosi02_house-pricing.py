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
# Importing Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
import os

# hide warnings
import warnings
warnings.filterwarnings('ignore')
# reading the train dataset
hp = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
hp.head()
# reading the test dataset
hp_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
hp_test.head()
# summary of the train dataset: 1460 rows, 81 columns, has null values
print(hp.info())
# Summary of test dataset
print(hp_test.info())
# Checking for null values in train set
hp.isnull().sum().sort_values(ascending=False).head(20)
# Checking for null values in test set
hp_test.isnull().sum().sort_values(ascending=False).head(40)
# columns with attributes like Pool, Fence etc. marked as NaN indicate the absence of these features.
attributes_with_na = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2','MasVnrType']
# replace 'NaN' with 'None' in these columns
for col in attributes_with_na:
    hp[col].fillna('None',inplace=True)
    hp_test[col].fillna('None',inplace=True)
# Checking for null values in train set
hp.isnull().sum().sort_values(ascending=False)
# Checking for null values in test set
hp_test.isnull().sum().sort_values(ascending=False).head(20)
# Imputing missing values in 'Lot Frontage' by median
hp['LotFrontage'].fillna(hp['LotFrontage'].median(),inplace=True)
hp_test['LotFrontage'].fillna(hp_test['LotFrontage'].median(),inplace=True)
# Imputing missing values in 'MasVnrType' by None
hp_test['MasVnrType'].fillna('None',inplace=True) 
# Imputing missing values in 'MasVnrArea' by 0 as None in 'MasVnrType' implies no masonry
hp['MasVnrArea'].fillna(0,inplace=True)
hp_test['MasVnrArea'].fillna(0,inplace=True)
# Imputing missing values by Median in continuous variable and by Mode in categorical variable

hp_test['GarageCars'].fillna(hp_test['GarageCars'].median(),inplace=True)
hp_test['GarageArea'].fillna(hp_test['GarageArea'].median(),inplace=True)
hp_test['KitchenQual'].fillna('TA',inplace=True)
hp_test['Exterior1st'].fillna('VinylSd',inplace=True)
hp_test['SaleType'].fillna('WD',inplace=True)
hp_test['TotalBsmtSF'].fillna(hp_test['TotalBsmtSF'].median(),inplace=True)
hp_test['BsmtUnfSF'].fillna(hp_test['BsmtUnfSF'].median(),inplace=True)
hp_test['BsmtFinSF1'].fillna(hp_test['BsmtFinSF1'].median(),inplace=True)
hp_test['BsmtFinSF2'].fillna(hp_test['BsmtFinSF2'].median(),inplace=True)
hp_test['Exterior2nd'].fillna('VinylSd',inplace=True)
hp_test['BsmtFullBath'].fillna(hp_test['BsmtFullBath'].median(),inplace=True)
hp_test['BsmtHalfBath'].fillna(hp_test['BsmtHalfBath'].median(),inplace=True)
hp_test['Functional'].fillna('Typ',inplace=True)
hp_test['Utilities'].fillna('AllPub',inplace=True)
hp_test['MSZoning'].fillna('RL',inplace=True)

hp['Electrical'].fillna('SBrkr',inplace=True)
# Checking for null values in train set
hp.isnull().sum().sort_values(ascending=False)
# Checking for null values in test set
hp_test.isnull().sum().sort_values(ascending=False).head(20)
hp.head()
# Dropping Id column
hp.drop('Id', 1, inplace = True)
hp_test_new = hp_test.drop('Id', 1)
# Function to check skewness in categorical columns
def chk_skewness(col):
    print(col)
    print(hp[col].value_counts(dropna=False,normalize=True))
    print('')
# Creating a list of all categorical variables
cat_var = [key for key in dict(hp.dtypes)
             if dict(hp.dtypes)[key] in ['object'] ] # Categorical Varible
cat_var
for col in cat_var:
    chk_skewness(col)
# dropping columns with more than 90% skewness
cat_col_with_skewness = ['MiscFeature','PoolQC','PavedDrive','GarageCond','Functional','Electrical','CentralAir','Heating','RoofMatl','Condition2','LandSlope','Utilities','Alley','Street']
hp.drop(cat_col_with_skewness,axis=1,inplace=True)
hp_test_new.drop(cat_col_with_skewness,axis=1,inplace=True)
hp.MSZoning.value_counts(normalize=True).plot.barh()
hp.LotShape.value_counts(normalize=True).plot.barh()
hp.LotConfig.value_counts(normalize=True).plot.barh()
hp.Neighborhood.value_counts(normalize=True).plot.barh()
hp.BldgType.value_counts(normalize=True).plot.barh()
hp.HouseStyle.value_counts(normalize=True).plot.barh()
hp.OverallQual.value_counts(normalize=True).plot.barh()
hp.OverallCond.value_counts(normalize=True).plot.barh()
hp.Foundation.value_counts(normalize=True).plot.barh()
hp.SaleType.value_counts(normalize=True).plot.barh()
hp.SaleCondition.value_counts(normalize=True).plot.barh()
plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
plt.scatter(hp.LotArea,hp.SalePrice)
plt.subplot(2,3,2)
plt.scatter(hp.TotalBsmtSF,hp.SalePrice)
plt.subplot(2,3,3)
plt.scatter(hp['1stFlrSF'],hp.SalePrice)
plt.subplot(2,3,4)
plt.scatter(hp['GarageArea'],hp.SalePrice)
plt.subplot(2,3,5)
plt.scatter(hp['GrLivArea'],hp.SalePrice)
plt.subplot(2,3,6)
plt.scatter(hp['WoodDeckSF'],hp.SalePrice)
# Correlation Heat Map
corr_matrix = hp.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corr_matrix, vmax=0.9, square=True)
import datetime
now = datetime.datetime.now()
#Imputing missing values in GarageYrBlt with current year so that age will be 0
hp['GarageYrBlt'].fillna(str(now.year),inplace=True)
hp_test_new['GarageYrBlt'].fillna(str(now.year),inplace=True)
# Changing datatype of GarageYrBlt
hp['GarageYrBlt']=hp['GarageYrBlt'].astype(int)
hp_test_new['GarageYrBlt']=hp_test_new['GarageYrBlt'].astype(int)
#Changing datatype of MoSold
hp['MoSold']=hp['MoSold'].astype('object')
hp_test_new['MoSold']=hp_test_new['MoSold'].astype('object')
#Converting Year variables to Age
hp['HouseAge'] = int(now.year) - hp['YearBuilt']
hp['HouseRemodelAge'] = int(now.year) - hp['YearRemodAdd']
hp['GarageAge'] = int(now.year) - hp['GarageYrBlt']
hp['SoldAge'] = int(now.year) - hp['YrSold']

hp_test_new['HouseAge'] = int(now.year) - hp_test_new['YearBuilt']
hp_test_new['HouseRemodelAge'] = int(now.year) - hp_test_new['YearRemodAdd']
hp_test_new['GarageAge'] = int(now.year) - hp_test_new['GarageYrBlt']
hp_test_new['SoldAge'] = int(now.year) - hp_test_new['YrSold']
# Dropping original Year Variables
hp = hp.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'],axis=1)
hp_test_new = hp_test_new.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'],axis=1)
# Creating a list of all numerical columns
num_cols=hp.select_dtypes(include=['float64', 'int64','int32']).columns
num_cols_test=hp_test_new.select_dtypes(include=['float64', 'int64','int32']).columns
num_cols
# Combining train and test set for Encoding and Scaling the variables
hp_comb = pd.concat([hp,hp_test_new])
hp_comb.info()
# Checking for null values in test set
hp_comb.isnull().sum().sort_values(ascending=False)
hp_comb[['ExterQual','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
            'HeatingQC','KitchenQual','GarageFinish','GarageQual','FireplaceQu','Fence',
             'ExterCond','LotShape']].head()
hp_comb['ExterQual'] = hp_comb.ExterQual.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
hp_comb['BsmtQual'] = hp_comb.BsmtQual.map({'NA':0,'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
hp_comb['BsmtCond'] = hp_comb.BsmtCond.map({'NA':0,'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
hp_comb['BsmtExposure'] = hp_comb.BsmtExposure.map({'NA':0,'None':0,'No':1,'Mn':2,'Av':3,'Gd':4})
hp_comb['BsmtFinType1'] = hp_comb.BsmtFinType1.map({'NA':0,'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
hp_comb['BsmtFinType2'] = hp_comb.BsmtFinType2.map({'NA':0,'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
hp_comb['HeatingQC'] = hp_comb.HeatingQC.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
hp_comb['KitchenQual'] = hp_comb.KitchenQual.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
hp_comb['GarageFinish'] = hp_comb.GarageFinish.map({'NA':0,'None':0,'Unf':1,'RFn':2,'Fin':3})
hp_comb['GarageQual'] = hp_comb.GarageQual.map({'NA':0,'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
hp_comb['FireplaceQu'] = hp_comb.FireplaceQu.map({'NA':0,'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
hp_comb['Fence'] = hp_comb.Fence.map({'None':0,'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4})
hp_comb['ExterCond'] = hp_comb.ExterCond.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
hp_comb['LotShape'] = hp_comb.LotShape.map({'IR1':0,'IR2':1,'IR3':2,'Reg':3})
dummy_col = pd.get_dummies(hp_comb[['MSZoning','LandContour','LotConfig','Neighborhood','Condition1','BldgType',
             'HouseStyle','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','Foundation',
             'GarageType','MoSold','SaleType','SaleCondition']],
                           drop_first=True)
print(dummy_col.shape)

hp_comb = pd.concat([hp_comb,dummy_col],axis=1)

hp_comb = hp_comb.drop(['MSZoning','LandContour','LotConfig','Neighborhood','Condition1','BldgType',
             'HouseStyle','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','Foundation',
             'GarageType','MoSold','SaleType','SaleCondition'],axis=1)
hp_comb.head()
sns.distplot(hp_comb['SalePrice']).set_title("Distribution of SalePrice")
hp_comb["SalePrice"] = np.log1p(hp_comb["SalePrice"])
sns.distplot(hp_comb['SalePrice']).set_title("Distribution of SalePrice")
df_train = hp_comb[0:1460]
df_train.info()
df_test = hp_comb[1460:]
df_test.info()
df_test.drop(['SalePrice'],axis=1,inplace=True)
df_test.head()
# Splitting the data into train and test
from sklearn.model_selection import train_test_split

hp_train, hp_test = train_test_split(df_train, train_size=0.8, test_size=0.2, random_state=42)
#scaling numeric columns

from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()
hp_train[num_cols_test] = scaler1.fit_transform(hp_train[num_cols_test])
hp_test[num_cols_test] = scaler1.transform(hp_test[num_cols_test])
df_test[num_cols_test] = scaler1.transform(df_test[num_cols_test])
hp_train.head()
y_train = hp_train.pop('SalePrice')
X_train = hp_train


y_test = hp_test.pop('SalePrice')
X_test = hp_test
lm  = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm,50)
rfe.fit(X_train,y_train)

rfe_scores = pd.DataFrame(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))
rfe_scores.columns = ['Column_Names','Status','Rank']

rfe_sel_columns = list(rfe_scores[rfe_scores.Status==True].Column_Names)

X_train1 = X_train[rfe_sel_columns]
X_test1 = X_test[rfe_sel_columns]
# Using GridSearchCV to find optimal value of alpha
folds = KFold(n_splits=10,shuffle=True,random_state=42)

hyper_param = {'alpha':[0.00001,0.0001,0.001, 0.01, 0.1,1.0, 5.0]}

model = Lasso()

model_cv = GridSearchCV(estimator = model,
                        param_grid=hyper_param,
                        scoring='neg_mean_squared_error',
                        cv=folds,
                        verbose=1,
                        return_train_score=True
                       )

model_cv.fit(X_train1,y_train)
cv_result_l = pd.DataFrame(model_cv.cv_results_)
cv_result_l['param_alpha'] = cv_result_l['param_alpha'].astype('float32')
cv_result_l.head()
plt.figure(figsize=(16,8))
plt.plot(cv_result_l['param_alpha'],cv_result_l['mean_train_score'])
plt.plot(cv_result_l['param_alpha'],cv_result_l['mean_test_score'])
plt.xscale('log')
plt.ylabel('MSE Score')
plt.xlabel('Alpha')
plt.legend(['train','test'])
plt.show()
# Checking the best parameter(Alpha value)
model_cv.best_params_
lasso = Lasso(alpha=0.0001)
lasso.fit(X_train1,y_train)

y_train_pred = lasso.predict(X_train1)
y_test_pred = lasso.predict(X_test1)

print('Train Set MSE : ',np.sqrt(mean_squared_error(y_true=y_train,y_pred=y_train_pred)))
print('Test Set MSE : ',np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_test_pred)))

print('Train Set R2 Score : ',r2_score(y_true=y_train,y_pred=y_train_pred))
print('Test Set R2 Score : ',r2_score(y_true=y_test,y_pred=y_test_pred))
df_test1 = df_test[rfe_sel_columns]
test_pred = lasso.predict(df_test1)
test_pred
import xgboost
from xgboost import plot_importance
#for tuning parameters
#parameters_for_testing = {
#    'colsample_bytree':[0.3,0.4,0.5,0.6],
#    'gamma':[0,0.03,0.05],
#    'min_child_weight':[1,1.3,1.5,1.6,1.8],
#    'learning_rate':[0.001,0.007,0.01,0.02],
#    'max_depth':[3,4,5],
#    'n_estimators':[10000],
#    'reg_alpha':[0.6,0.7,0.75,0.8],
#    'reg_lambda':[1e-2, 0.45,0.55,0.6],
#    'subsample':[0.4,0.5,0.6,0.7]
#}


#xgb_model = xgboost.XGBRegressor(tree_method='gpu_exact',random_state=42)
#gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=-1,iid=False, verbose=10,scoring='neg_mean_squared_error')
#gsearch1.fit(X_train,y_train)
#print (gsearch1.grid_scores_)
#print('best params')
#print (gsearch1.best_params_)
#print('best score')
#print (gsearch1.best_score_)
xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.7,
                 reg_lambda=0.55,
                 subsample=0.5,
                 seed=42)
xgb_model.fit(X_train,y_train)
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

print('Train Set MSE : ',np.sqrt(mean_squared_error(y_true=y_train,y_pred=y_train_pred)))
print('Test Set MSE : ',np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_test_pred)))

print('Train Set R2 Score : ',r2_score(y_true=y_train,y_pred=y_train_pred))
print('Test Set R2 Score : ',r2_score(y_true=y_test,y_pred=y_test_pred))
df_train[num_cols_test] = scaler1.fit_transform(df_train[num_cols_test])
final_train = df_train
final_train.head()
f_y_train =final_train.pop('SalePrice')
f_x_train = final_train
xgb_model.fit(f_x_train,f_y_train)
test_pred = xgb_model.predict(df_test)
test_pred
# Converting test_pred to a dataframe which is an array
test_pred_1 = pd.DataFrame(test_pred)
# Renaming the column 
test_pred_1= test_pred_1.rename(columns={ 0 : 'SalePrice'})
test_pred_1.head()
test_pred_1["SalePrice"] = np.expm1(test_pred_1["SalePrice"])
test_pred_1.head()
#test_pred_1.to_csv('Submission5.csv')
filename = 'final_submission1.csv'
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
pd.DataFrame({'Id': test.Id, 'SalePrice': test_pred_1.SalePrice}).to_csv(filename, index=False)