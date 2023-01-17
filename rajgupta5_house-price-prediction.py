import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

import numpy as np



# allow plots to appear directly in the notebook

%matplotlib inline

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import os



from sklearn.model_selection import RandomizedSearchCV
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
base_path='/kaggle/input/house-prices-advanced-regression-techniques/'
train = pd.read_csv(base_path + 'train.csv')

test = pd.read_csv(base_path + 'test.csv')
print("Original Shape of Train and Test before starting EDA")

print(train.shape)

print(test.shape)
train.head(5)
test.head(5)
sns.distplot(train.SalePrice, kde=True)
train.SalePrice = np.log(train.SalePrice)
sns.distplot(train.SalePrice, kde=True)
test_Id = test.Id

train_Id = train.Id

train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
train_index = train.shape[0]

test_index = test.shape[0]

dataset = pd.concat((train, test)).reset_index(drop=True)

print("all_data size is : {}".format(dataset.shape))
dataset_na = (dataset.isnull().sum() / len(dataset)) * 100

dataset_na = dataset_na.drop(dataset_na[dataset_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :dataset_na})

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=dataset_na.index, y=dataset_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False)

dataset.drop(['PoolQC','Fence','MiscFeature', 'Alley'],axis=1,inplace=True)

dataset.drop(['GarageYrBlt'],axis=1,inplace=True)
def missing_imputation(x, stats = 'mean'):

    if (x.dtypes == 'float64') | (x.dtypes == 'int64'):

        x = x.fillna(x.mean()) if stats == 'mean' else x.fillna(x.median())

    else:

        x = x.fillna(x.mode())

    return x
# all_data = all_data.apply(missing_imputation)
dataset['LotFrontage']=dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())

dataset['BsmtHalfBath']= dataset['BsmtHalfBath'].fillna(dataset['BsmtHalfBath'].mean())

dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(dataset['BsmtFullBath'].mean())

dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].mean())

dataset['GarageCars'] = dataset['GarageCars'].fillna(dataset['GarageCars'].mean())

dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean())
dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(dataset['BsmtUnfSF'].mean())

dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(dataset['BsmtFinSF1'].mean())

dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(dataset['BsmtFinSF2'].mean())
dataset['BsmtCond']=dataset['BsmtCond'].fillna(dataset['BsmtCond'].mode()[0])

dataset['BsmtQual']=dataset['BsmtQual'].fillna(dataset['BsmtQual'].mode()[0])
dataset['GarageFinish']=dataset['GarageFinish'].fillna(dataset['GarageFinish'].mode()[0])

dataset['GarageQual']=dataset['GarageQual'].fillna(dataset['GarageQual'].mode()[0])

dataset['GarageCond']=dataset['GarageCond'].fillna(dataset['GarageCond'].mode()[0])
dataset['MasVnrType']=dataset['MasVnrType'].fillna(dataset['MasVnrType'].mode()[0])

dataset['MasVnrArea']=dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mode()[0])
dataset['BsmtExposure']=dataset['BsmtExposure'].fillna(dataset['BsmtExposure'].mode()[0])

dataset['BsmtFinType2']=dataset['BsmtFinType2'].fillna(dataset['BsmtFinType2'].mode()[0])

dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(dataset['BsmtFinType1'].mode()[0])

dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode()[0])

dataset['Utilities'] = dataset['Utilities'].fillna(dataset['Utilities'].mode()[0])
dataset['Functional'] = dataset['Functional'].fillna(dataset['Functional'].mode()[0])

dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].mode()[0])

dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].mode()[0])

dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].mode()[0])

dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].mode()[0])

dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode()[0])
dataset['FireplaceQu']=dataset['FireplaceQu'].fillna(dataset['FireplaceQu'].mode()[0])

dataset['GarageType']=dataset['GarageType'].fillna(dataset['GarageType'].mode()[0])
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
plt.figure(figsize=[30,15])

sns.heatmap(dataset.corr(), annot=True)

plt.ylim(40,0)
dataset.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'], axis=1, inplace=True)

dataset_cat = dataset.loc[:, (dataset.dtypes == 'object')]
dataset_cat.columns
def category_onehot_multcols(multcolumns):

    df_final=final_df

    i=0

    for fields in multcolumns:

        

        print(fields)

        df1=pd.get_dummies(final_df[fields],drop_first=True)

        

        final_df.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:

            

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([final_df,df_final],axis=1)

        

    return df_final
final_df=dataset
final_df.shape
dataset = category_onehot_multcols(dataset_cat.columns)
dataset.shape
dataset =dataset.loc[:,~dataset.columns.duplicated()]

dataset.head(2)
train = dataset[:train_index]

test = dataset[train_index:]
test.drop('SalePrice', axis=1, inplace=True)
print(train.shape)

print(test.shape)
train.head()
X_train = train.drop('SalePrice', axis=1)

y_train = train.SalePrice
print(X_train.shape)

print(test.shape)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

test=sc.transform(test)
import xgboost

regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
## Hyper Parameter Optimization





n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
# Set up the random search with 4-fold cross validation

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.1, max_delta_step=0,

             max_depth=2, min_child_weight=1, missing=None, n_estimators=900,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)
regressor.fit(X_train,y_train)
import pickle

filename = 'finalized_model.pkl'

pickle.dump(regressor, open(filename, 'wb'))

test_SalePrice = regressor.predict(test)

test_SalePrice_pred=np.exp(test_SalePrice)
sub = pd.DataFrame()

sub['Id'] = test_Id

sub['SalePrice'] = test_SalePrice_pred

sub.to_csv('submission.csv',index=False)
sub.head()