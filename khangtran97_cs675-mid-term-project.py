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
# numpy and pandas for data manipulation

import numpy as np

import pandas as pd 

from sklearn.preprocessing import LabelEncoder

import os 

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew

from scipy.stats.stats import pearsonr
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
test_df['SalePrice'] = np.nan

test_df.head()
print(train_df.shape, test_df.shape)
train_df = train_df.drop(['Id'], axis = 1)

test_df = test_df.drop(['Id'], axis = 1)

target = train_df['SalePrice']

all_data = pd.concat([train_df, test_df], ignore_index = True)

all_data.head()
target.hist()
print(target.skew())

target_log = np.log1p(np.array(target))
pd.DataFrame(target_log).hist()
# Function to calculate missing values by column# Funct 

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values = missing_values_table(all_data)

missing_values
all_data.dtypes.value_counts()
all_data.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# #log transform skewed numeric features:

# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# skewed_feats = train_df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

# skewed_feats = skewed_feats[skewed_feats > 0.75]

# skewed_feats = skewed_feats.index



# all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data.info()
all_data[['PoolQC', 'PoolArea']]
all_data['PoolQC'] = all_data['PoolQC'].fillna('NA')

print(all_data['PoolQC'].isnull().sum())

all_data.head()
all_data.loc[(all_data['PoolQC'] == 'NA') & (all_data['PoolArea'] != 0)]
all_data.at[2420, 'PoolQC'] = 'Fa'

all_data.at[2599, 'PoolQC'] = 'Fa'

all_data.at[2503, 'PoolQC'] = 'Fa'
maps = {

    'Ex': 5,

    'Gd': 4,

    'TA': 3,

    'Fa': 2,

    'NA': 1

}

all_data['PoolQC'] = all_data['PoolQC'].replace(maps)

all_data.head()
all_data['MiscFeature'].describe()

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('NA')
all_data['Alley'].describe()

all_data['Alley'] = all_data['Alley'].fillna('NA')
all_data['Fence'].describe()

all_data['Fence'] = all_data['Fence'].fillna('NA')
all_data['FireplaceQu'].describe()
all_data[['Fireplaces','FireplaceQu']][all_data['FireplaceQu'].isnull() == True]
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('NA')
maps = {

    'NA':1,

    'Po':2,

    'Fa':3,

    'TA':4,

    'Gd':5,

    'Ex':6

}

all_data['FireplaceQu'] = all_data['FireplaceQu'].replace(maps)
plt.scatter(all_data['LotFrontage'], all_data['SalePrice'])
all_data[all_data['LotFrontage'].isnull() == True].SalePrice.median()

all_data['LotFrontage'] = all_data['LotFrontage'].fillna(all_data['LotFrontage'].median())
all_data['LotFrontage'].median()
all_data[['GarageCond','GarageYrBlt','GarageFinish','GarageQual','GarageType','GarageArea','GarageCars']]
all_data['GarageCond'] = all_data['GarageCond'].fillna('NA').replace(maps)

all_data['GarageQual'] = all_data['GarageQual'].fillna('NA').replace(maps)

all_data[['GarageCond','GarageYrBlt','GarageFinish','GarageQual','GarageType','GarageArea','GarageCars']]
mapp = {

    'Fin': 4,

    'RFn': 3,

    'Unf': 2,

    'NA': 1

}

all_data['GarageFinish'] = all_data['GarageFinish'].fillna('NA').replace(mapp)

all_data['GarageType'] = all_data['GarageType'].fillna('NA')

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(-1)
all_data[['GarageCond','GarageYrBlt','GarageFinish','GarageQual','GarageType','GarageArea','GarageCars']][all_data['GarageArea'].isnull() == True]

all_data['GarageArea'] = all_data['GarageArea'].fillna(0.0)

all_data['GarageCars'] = all_data['GarageCars'].fillna(0.0)
train_df[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtHalfBath','BsmtFullBath']]
all_data[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtHalfBath','BsmtFullBath']]
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0.0)

all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0.0)
all_data[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtHalfBath','BsmtFullBath']][all_data['BsmtExposure'].isnull() == True]
mapp = {

    'Gd':5,

    'Av':4,

    'Mn':3,

    'No':2,

    'NA':1

}

all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('NA').replace(mapp)

all_data['BsmtCond'] = all_data['BsmtCond'].fillna('NA').replace(maps)

all_data['BsmtQual'] = all_data['BsmtQual'].fillna('NA').replace(maps)
mapp = {

    'GLQ':7,

    'ALQ':6,

    'BLQ':5,

    'Rec':4,

    'LwQ':3,

    'Unf':2,

    'NA':1

}

all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna('NA').replace(mapp)

all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna('NA').replace(mapp)
missing_values = missing_values_table(all_data)

missing_values
all_data[['MasVnrType','MasVnrArea']][all_data['MasVnrType'].isnull() == True]

all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0.0)

all_data.at[2610, 'MasVnrType'] = 'Stone'
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0.0)

all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0.0)

all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0.0)

all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0.0)
missing_values = missing_values_table(all_data)

missing_values
all_data['SaleType'] = all_data.SaleType.fillna('Oth')

all_data['KitchenQual'] = all_data.KitchenQual.fillna('Oth')
missing_values = missing_values_table(all_data)

missing_values
all_data[all_data['Utilities'].isnull() == True].Functional
for col in all_data.columns:

    if((all_data[col].isnull().sum() > 0) & (col != 'SalePrice')):

        all_data[col] = all_data[col].fillna('NA')
all_data.head()
all_data['totalBathroom'] = 0.5*all_data['HalfBath'] + all_data['FullBath'] + 0.5*all_data['BsmtHalfBath'] + all_data['FullBath']

all_data.head()
all_data['age'] = all_data['YrSold'] - all_data['YearRemodAdd']

ismodeled = []

deltaYr = all_data['YearRemodAdd'] - all_data['YearBuilt']

print(type(deltaYr))
ismodeled = []

for i in deltaYr:

    if i == 0:

        ismodeled.append(1)

    else:

        ismodeled.append(0)

all_data['IsModeled'] = ismodeled

all_data.head()
deltaYr = all_data['YrSold'] - all_data['YearBuilt']

isnew = []

for i in deltaYr:

    if i == 0:

        isnew.append(1)

    else:

        isnew.append(0)

all_data['IsNew'] = isnew

all_data.head()
all_data['totalSquareFeet'] = all_data['GrLivArea'] + all_data['TotalBsmtSF']

all_data.head()
#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data.head()
linearFeat = []

cor = all_data.corr()['SalePrice'].sort_values(ascending=False)[all_data.corr()['SalePrice'] > 0.6].index

for i in cor:

    if i != 'SalePrice':

        linearFeat.append(i)
target_log
all_data = all_data.drop('SalePrice', axis = 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

for i in numeric_feats:

    all_data[i] = scaler.fit_transform(np.array(all_data[i]).reshape(-1, 1))

all_data.head()

# all_data[numeric_feats] = np.log1p(all_data[skewed_feats])
train = all_data[:train_df.shape[0]]

test = all_data[train_df.shape[0]:]
train.head()
train.dtypes.value_counts()
# Create a label encoder object

le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in all_data:

    if all_data[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(all_data[col].unique())) <= 2:

            # Train on the training data

            le.fit(all_data[col])

            # Transform both training and testing data

            all_data[col] = le.transform(all_data[col])

            le_count += 1

            

print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables

all_data = pd.get_dummies(all_data)

print('Testing Features shape: ', all_data.shape)
# from sklearn.decomposition import KernelPCA 

# from sklearn.decomposition import PCA

# kpca = PCA(n_components = 40) 

# all_data_pca = kpca.fit_transform(all_data)

# all_data_pca = pd.DataFrame(all_data_pca)

# all_data_pca.head()
# all_data_pca.shape
all_data['Id'] = all_data.index

all_data.head()
train = all_data[:train_df.shape[0]]

test = all_data[train_df.shape[0]:]
target = pd.DataFrame({'SalePrice': target_log})

target.head()
linearFeat.append('Id')

train = train[linearFeat]

test = test[linearFeat]
train.head()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import math  

from sklearn.model_selection import KFold, StratifiedKFold



score = []

predict_val = pd.DataFrame(test['Id'])



skf = KFold(n_splits = 5, shuffle=True, random_state=123)

skf.get_n_splits(train, target_log)

oof_lr_df = pd.DataFrame()

predictions = pd.DataFrame(test['Id'])

x_test = test.drop(['Id'], axis = 1)



for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target_log)):

    x_train, y_train = train.iloc[trn_idx], target.iloc[trn_idx]['SalePrice']

    x_valid, y_valid = train.iloc[val_idx], target.iloc[val_idx]['SalePrice']

    index = x_valid['Id']

    yp = 0

    x_train = x_train.drop(['Id'], axis = 1)

    x_valid = x_valid.drop(['Id'], axis = 1)

    lr = LinearRegression()

    lr.fit(x_train,y_train)

    score.append(math.sqrt(mean_squared_error(lr.predict(x_valid), y_valid)))

    yp += lr.predict(x_test)

    fold_pred = pd.DataFrame({'ID': index,

                              'label':lr.predict(x_valid)})

    oof_lr_df = pd.concat([oof_lr_df, fold_pred], axis=0)

    predictions['fold{}'.format(fold+1)] = yp
score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

oof_lr_df = oof_lr_df.sort_values('ID')

oof_lr_df.head(10)
lr_predict = pd.DataFrame()

lr_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5'])/5

lr_predict.head()
train = all_data[:train_df.shape[0]]

test = all_data[train_df.shape[0]:]
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

import math  

from sklearn.model_selection import KFold, StratifiedKFold



score = []

predict_val = pd.DataFrame(test['Id'])



skf = KFold(n_splits = 5, shuffle=True, random_state=123)

skf.get_n_splits(train, target_log)

oof_rfr_df = pd.DataFrame()

predictions = pd.DataFrame(test['Id'])

x_test = test.drop(['Id'], axis = 1)



for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target_log)):

    x_train, y_train = train.iloc[trn_idx], target.iloc[trn_idx]['SalePrice']

    x_valid, y_valid = train.iloc[val_idx], target.iloc[val_idx]['SalePrice']

    index = x_valid['Id']

    yp = 0

    x_train = x_train.drop(['Id'], axis = 1)

    x_valid = x_valid.drop(['Id'], axis = 1)

    rfr = RandomForestRegressor(random_state=1, max_depth=10)

    rfr.fit(x_train,y_train)

    score.append(math.sqrt(mean_squared_error(rfr.predict(x_valid), y_valid)))

    yp += rfr.predict(x_test)

    fold_pred = pd.DataFrame({'ID': index,

                              'label':rfr.predict(x_valid)})

    oof_rfr_df = pd.concat([oof_rfr_df, fold_pred], axis=0)

    predictions['fold{}'.format(fold+1)] = yp
score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

oof_rfr_df = oof_lr_df.sort_values('ID')

oof_rfr_df.head(10)
rfr_predict = pd.DataFrame()

rfr_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5'])/5

rfr_predict.head()
train = all_data[:train_df.shape[0]]

test = all_data[train_df.shape[0]:]
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error

import math  

from sklearn.model_selection import KFold, StratifiedKFold



score = []

predict_val = pd.DataFrame(test['Id'])



skf = KFold(n_splits = 5, shuffle=True, random_state=123)

skf.get_n_splits(train, target_log)

oof_svr_df = pd.DataFrame()

predictions = pd.DataFrame(test['Id'])

x_test = test.drop(['Id'], axis = 1)



for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target_log)):

    x_train, y_train = train.iloc[trn_idx], target.iloc[trn_idx]['SalePrice']

    x_valid, y_valid = train.iloc[val_idx], target.iloc[val_idx]['SalePrice']

    index = x_valid['Id']

    yp = 0

    x_train = x_train.drop(['Id'], axis = 1)

    x_valid = x_valid.drop(['Id'], axis = 1)

    svm_reg = SVR(epsilon=0.1)

    svm_reg.fit(x_train,y_train)

    score.append(math.sqrt(mean_squared_error(svm_reg.predict(x_valid), y_valid)))

    yp += svm_reg.predict(x_test)

    fold_pred = pd.DataFrame({'ID': index,

                              'label':svm_reg.predict(x_valid)})

    oof_svr_df = pd.concat([oof_svr_df, fold_pred], axis=0)

    predictions['fold{}'.format(fold+1)] = yp
score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

oof_svr_df = oof_svr_df.sort_values('ID')

oof_svr_df.head(10)
svr_predict = pd.DataFrame()

svr_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5'])/5

svr_predict.head()
train = all_data[:train_df.shape[0]]

test = all_data[train_df.shape[0]:]
from sklearn.metrics import mean_squared_error

import math  

from xgboost import XGBRegressor

from sklearn.model_selection import KFold, StratifiedKFold





score = []

predict_val = pd.DataFrame(test['Id'])



skf = KFold(n_splits = 5, shuffle=True, random_state=123)

skf.get_n_splits(train, target_log)

oof_xgb_df = pd.DataFrame()

predictions = pd.DataFrame(test['Id'])

x_test = test.drop(['Id'], axis = 1)



for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target_log)):

    x_train, y_train = train.iloc[trn_idx], target.iloc[trn_idx]['SalePrice']

    x_valid, y_valid = train.iloc[val_idx], target.iloc[val_idx]['SalePrice']

    index = x_valid['Id']

    yp = 0

    x_train = x_train.drop(['Id'], axis = 1)

    x_valid = x_valid.drop(['Id'], axis = 1)

    xgb = XGBRegressor(max_depth = 8, n_estimators = 2000, n_jobs = 16, random_state = 4, subsample = 0.8, colsample_bytree = 0.7, max_bin = 16, tree_method = 'gpu_hist', gpu_id = 0)

    xgb.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)], eval_metric = ['rmse'], early_stopping_rounds = 70, verbose = 200)

    score.append(math.sqrt(mean_squared_error(svm_reg.predict(x_valid), y_valid)))

    yp += xgb.predict(x_test)

    fold_pred = pd.DataFrame({'ID': index,

                              'label':xgb.predict(x_valid)})

    oof_xgb_df = pd.concat([oof_xgb_df, fold_pred], axis=0)

    predictions['fold{}'.format(fold+1)] = yp
score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

oof_xgb_df = oof_xgb_df.sort_values('ID')

oof_xgb_df.head(10)
xgb_predict = pd.DataFrame()

xgb_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5'])/5

xgb_predict.head()
oof_data = None

oof_data = pd.DataFrame({ 'linearReg': oof_lr_df['label'],

                          'SVMReg': oof_svr_df['label'],

                          'RandomForestReg':  oof_rfr_df['label'],

                          'XgbRegression': oof_xgb_df['label'],

                          'label': target['SalePrice']

})

oof_data.head()
oof_test = pd.DataFrame({ 'linearReg': lr_predict['predict'],

                          'SVMReg': svr_predict['predict'],

                          'RandomForestReg':  rfr_predict['predict'],

                          'XgbRegression': xgb_predict['predict']

})

oof_test.head()
oof_data.corr()
lr = LinearRegression()

lr.fit(oof_data.drop(['label'],axis=1),oof_data['label'])

predict = np.exp(lr.predict(oof_test))-1
mysubmit = pd.DataFrame({'Id': test['Id']+1,

                         'SalePrice': predict})

mysubmit.head()
mysubmit.to_csv('mysubmit.csv', index=False)
# from sklearn.ensemble import RandomForestRegressor



# model = RandomForestRegressor(random_state=1, max_depth=10)



# model.fit(train,target_log)
# dict(reversed(sorted(zip(model.feature_importances_, zip(train.columns.values, train_df.dtypes.values)))))
# y_pred = model.predict(train)
# import math

# from sklearn.metrics import mean_squared_error

# print(math.sqrt(mean_squared_error(target_log, y_pred)))
# train['OverallQual'].hist()
# test['OverallQual'].hist()
# from sklearn.model_selection import train_test_split

# x_train, x_valid, y_train, y_valid = train_test_split(train, target_log, random_state=20, stratify=train['OverallQual'])
# model = RandomForestRegressor(random_state=1, max_depth=5)



# model.fit(x_train,y_train)



# y_pred = model.predict(x_valid)



# import math

# from sklearn.metrics import mean_squared_error



# print(math.sqrt(mean_squared_error(y_valid, y_pred)))
# all_data.head()
# all_data.columns.sort_values(ascending=False)