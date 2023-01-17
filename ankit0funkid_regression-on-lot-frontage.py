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

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures



from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

np.set_printoptions(formatter={'float_kind':'{:0.2f}'.format})
train_df = pd.read_csv(r'../input/house-prices-advanced-regression-techniques/train.csv')
train_df.info()
plt.figure(figsize = (15,8))

sns.heatmap(train_df.isnull())
def null_details(data_df):

    rows = len(data_df)

    null_counts = data_df.isnull().sum()

    null_perc = (null_counts.values/rows)*100

    data_df_nulls = pd.DataFrame(null_counts,columns = ['Count'])

    data_df_nulls['Percentage'] = null_perc

    data_df_nulls = data_df_nulls[data_df_nulls['Count'] > 0]

    return data_df_nulls
null_details(train_df)
train_df_n = train_df.copy()
null_dict = {

    'Alley':'NA',

    'MasVnrType':'NA',

    'MasVnrArea':0,

    'BsmtQual':'NA',

    'BsmtCond':'NA',

    'BsmtExposure':'NA',

    'BsmtFinType1':'NA',

    'BsmtFinType2':'NA',

    'FireplaceQu':'NA',

    'GarageType':'NA',

    'GarageYrBlt':0,

    'GarageFinish':'NA',

    'GarageQual':'NA',

    'GarageCond':'NA',

    'PoolQC':'NA',

    'Fence':'NA',

    'MiscFeature':'NA'

}
for val in null_dict:

    train_df_n[val] = train_df_n[val].fillna(null_dict[val])
null_details(train_df_n)
def unCat_data_mapping(data_df,columns):

    colCat_maps = {}

    for col in columns:

        map_dict = {}

        for i,val in enumerate(train_df_n[col].unique()):

            map_dict[val] = i

#             print(col,end = ' : ')

#             print(map_dict)

        colCat_maps[col] = map_dict

        data_df[col] = data_df[col].map(map_dict)

    return data_df,colCat_maps
nomCat_cols = ['MSSubClass',

'MSZoning',

'Street',

'Alley',

'LotShape',

'LandContour',

'LotConfig',

'LandSlope',

'Neighborhood',

'Condition1',

'Condition2',

'BldgType',

'HouseStyle',

'RoofStyle',

'RoofMatl',

'Exterior1st',

'Exterior2nd',

'MasVnrType',

'Foundation',

'Heating',

'Electrical',

'GarageType',

'MiscFeature',

'SaleType',

'SaleCondition']
trainDF_cat = train_df_n.copy()
trainDF_cat,map_dicts = unCat_data_mapping(trainDF_cat,nomCat_cols)
# Mapping dictionaries

Utilities_dict = {'AllPub':4,'NoSewr':3,'NoSeWa':2,'ELO':1}

ExterQual_dict = {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,'NA':0}

ExterCond_dict = {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,'NA':0}

BsmtQual_dict = {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,'NA':0}

BsmtCond_dict = {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,'NA':0}

HeatingQC_dict = {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,'NA':0}

KitchenQual_dict = {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,'NA':0}

FireplaceQu_dict = {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,'NA':0}

GarageQual_dict = {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,'NA':0}

GarageCond_dict = {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,'NA':0}

PoolQC_dict = {'Ex':6,'Gd':5,'TA':4,'Fa':3,'Po':2,'NA':0}

BsmtExposure_dict = {'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0}

BsmtFinType1_dict = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}

BsmtFinType2_dict = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}

CentralAir_dict = {'Y':1,'N':0}

Functional_dict = {'Typ':8,'Min1':7,'Min2':6,'Mod':5,'Maj1':4,'Maj2':3,'Sev':2,'Sal':1}

GarageFinish_dict = {'Fin':3,'RFn':2,'Unf':1,'NA':0}

PavedDrive_dict = {'Y':2,'P':1,'N':0}

Fence_dict = {'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'NA':0}
# Adding to main mapping dict

map_dicts['Utilities'] = Utilities_dict

map_dicts['ExterQual'] = ExterQual_dict

map_dicts['ExterCond'] = ExterCond_dict

map_dicts['BsmtQual'] = BsmtQual_dict

map_dicts['BsmtCond'] = BsmtCond_dict

map_dicts['BsmtExposure'] = BsmtExposure_dict

map_dicts['BsmtFinType1'] = BsmtFinType1_dict

map_dicts['BsmtFinType2'] = BsmtFinType2_dict

map_dicts['HeatingQC'] = HeatingQC_dict

map_dicts['CentralAir'] = CentralAir_dict

map_dicts['KitchenQual'] = KitchenQual_dict

map_dicts['Functional'] = Functional_dict

map_dicts['FireplaceQu'] = FireplaceQu_dict

map_dicts['GarageFinish'] = GarageFinish_dict

map_dicts['GarageQual'] = GarageQual_dict

map_dicts['GarageCond'] = GarageCond_dict

map_dicts['PavedDrive'] = PavedDrive_dict

map_dicts['PoolQC'] = PoolQC_dict

map_dicts['Fence'] = Fence_dict
ordinal_cols = ['Utilities','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'

,'HeatingQC','CentralAir','KitchenQual','Functional','FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive'

,'PoolQC','Fence']
for col in ordinal_cols:

    trainDF_cat[col] = trainDF_cat[col].map(map_dicts[col])
plt.figure(figsize = (15,10))

sns.heatmap(trainDF_cat.isnull())
null_details(trainDF_cat)
fig = plt.figure(figsize=(15,8))

sns.lineplot(x = "LotFrontage",y = "LotArea",data = trainDF_cat)
lot_dims = trainDF_cat[~pd.isnull(trainDF_cat['LotFrontage'])]

lot_dims['LotLength'] = lot_dims['LotArea']/lot_dims['LotFrontage']

lot_dims[['LotArea','LotFrontage','LotShape','LotLength']].head()
fig, ax = plt.subplots(1,figsize=(15,8))

sns.distplot(lot_dims['LotLength'],ax = ax)

ax.xaxis.set_major_locator(plt.MaxNLocator(30))
fig = plt.figure(figsize=(10,20))

cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

sns.heatmap(lot_dims.corr().sort_values(by = ['LotFrontage'],ascending = False)[['LotFrontage']]

            ,vmin = -1,vmax = 1,

            cmap = cmap)
lotFront_df = trainDF_cat.copy()
null_details(lotFront_df)
lF_nnull_df = lotFront_df.dropna()

null_details(lF_nnull_df)
tgt_var = lF_nnull_df.LotFrontage



X_train, X_test, y_train, y_test = train_test_split(lF_nnull_df, tgt_var, test_size=0.2)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
fig = plt.figure(figsize=(15,8))

sns.distplot(lF_nnull_df[['LotFrontage']])
LotFront_Corr = lot_dims.corr().sort_values(by = ['LotFrontage'])[['LotFrontage']]

LotFront_Corr.drop(LotFront_Corr.tail(2).index,inplace = True)
indep_var = pd.concat([LotFront_Corr.head(3),LotFront_Corr.tail(3)]).index
x = X_train[indep_var]

y = y_train
poly_features = PolynomialFeatures(degree = 3)

x_poly = poly_features.fit_transform(x)
model = LinearRegression()

model.fit(x_poly,y)

y_poly_pred = model.predict(x_poly)
rmse = np.sqrt(mean_squared_error(y,y_poly_pred))

print("%.20f" % rmse)

r2 = r2_score(y,y_poly_pred)

print(r2)
fig = plt.figure(figsize=(15,8))

sns.distplot(y_poly_pred,color = 'Red',hist = False)

sns.distplot(y,color = 'blue',hist = False)
fig = plt.figure(figsize=(15,7))

sns.scatterplot(x = X_train['LotArea'].values,y = y_train)

sns.scatterplot(x = X_train['LotArea'].values,y = y_poly_pred,alpha = 0.2)
x_test_poly = poly_features.fit_transform(X_test[indep_var])

y_test_pred = model.predict(x_test_poly)
fig, ax = plt.subplots(1,figsize=(15,8))

sns.distplot(y_test,ax = ax,hist = False,kde_kws=dict(linewidth=5))

sns.distplot(y_test_pred,ax = ax,hist = False)
rmse = np.sqrt(mean_squared_error(y,y_poly_pred))

print("%.20f" % rmse)

r2 = r2_score(y,y_poly_pred)

print(r2)
fig = plt.figure(figsize=(15,8))

sns.scatterplot(x = X_test['LotArea'].values,y = y_test)

sns.scatterplot(x = X_test['LotArea'].values,y = y_test_pred,alpha = 0.5)
x = lotFront_df[pd.isnull(lotFront_df['LotFrontage'])][indep_var]

x_pred_poly = poly_features.fit_transform(x)

y_pred = model.predict(x_pred_poly)

x['LotFrontage'] = y_pred
fig, ax = plt.subplots(1,figsize=(15,5))

sns.distplot(x)
fig, ax = plt.subplots(1,figsize=(15,6))

sns.scatterplot(x = lotFront_df[pd.isnull(lotFront_df['LotFrontage'])]['LotArea'].values,y = y_pred)
neg_ind = x.loc[x['LotFrontage']<0].index

for idx in neg_ind:

    x.loc[idx,'LotFrontage'] = np.nan
fig, ax = plt.subplots(1,figsize=(15,6))

sns.scatterplot(y = x['LotFrontage'],x = x['LotArea'])
rep_indx = x.index

for idx in rep_indx:

    lotFront_df.loc[idx,'LotFrontage'] = x.loc[idx,'LotFrontage']
null_details(lotFront_df)
fig, ax = plt.subplots(1,figsize=(15,8))

sns.distplot(lotFront_df['LotFrontage'])
lotFront_df.to_csv('HousePrices_DataHandled.csv')