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

        from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

import string

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.ticker import MaxNLocator

import seaborn as sns



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def divide_df(all_data):

    # Returns divided dfs of training and test set

    return all_data.loc[:1459], all_data.loc[1460:].drop(['SalePrice'], axis=1)
iowa_file_train_path = '/kaggle/input/home-data-for-ml-course/train.csv'

iowa_file_test_path="/kaggle/input/home-data-for-ml-course/test.csv"

train=pd.read_csv(iowa_file_train_path)

test=pd.read_csv(iowa_file_test_path)

df_all=concat_df(train,test)
train.describe()
test.describe()
train_row=train.shape[0]

test_row=test.shape[0]

print('Number of Rows in Training Examples = {}'.format(train_row))

print('Number of Rows in Testing Examples = {}\n'.format(test_row))

print('Training X shape={}'.format(train.shape))

print('Training y shape={}'.format(train['SalePrice'].shape[0]))

print('Testing X shape={}'.format(test.shape))

print('Testing y shape={}'.format(test.shape[0]))
train.head()
test.head()
train.head(-1)
print(train.info())
print(test.info())
print(df_all.info())
drop_cols=['Alley','Fence','FireplaceQu','PoolQC','LotFrontage','MiscFeature' ]

df_all.drop(columns=drop_cols, inplace=True)
zero_cols = [

    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',

    'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'

]

for col in zero_cols:

    df_all[col].replace(np.nan, 0, inplace=True)
corrmat = df_all.corr()



top_corr_features = corrmat.index

plt.figure(figsize=(45,45))

#plot heat map

g=sns.heatmap(df_all[top_corr_features].corr(),annot=True,cmap="RdYlGn")
corrmat['SalePrice'].sort_values(ascending=False).head(20)
train,test=divide_df(df_all)
train.head()
test.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



features=[

'OverallQual',

'GrLivArea',      

#'ExterQual',      

'GarageCars',     

'GarageArea' ,     

'TotalBsmtSF',     

'1stFlrSF' ,       

'FullBath' ,       

'TotRmsAbvGrd',    

'YearBuilt' ,      

'YearRemodAdd',    

'MasVnrArea',      

'Fireplaces',      

'BsmtFinSF1'  ,    

'WoodDeckSF'  ,   

'2ndFlrSF',        

'OpenPorchSF',    

'HalfBath',        

'LotArea']

X=train[features]

y=train.SalePrice
train_X,train_y,val_X,val_y=train_test_split(X,y,random_state=1)
rf_model=RandomForestRegressor()

rf_model.fit(X,y)
test_data_path='/kaggle/input/home-data-for-ml-course/test.csv'

test_data = pd.read_csv(test_data_path)

test_X = test_data[features]

X_test = test_X.fillna(train_X.mean())

test_preds = rf_model.predict(X_test )

print(test_preds)



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission1.csv', index=False)
