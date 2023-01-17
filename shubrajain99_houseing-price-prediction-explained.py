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

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

iowa_file_test_path="../input/home-data-for-ml-course/test.csv"

dataset_train=pd.read_csv(iowa_file_path)

dataset_test=pd.read_csv(iowa_file_test_path)

df_all = concat_df(dataset_train, dataset_test)
print('Number of Training Examples = {}'.format(dataset_train.shape[0]))

print('Number of Test Examples = {}\n'.format(dataset_test.shape[0]))

print('Training X Shape = {}'.format(dataset_train.shape))

print('Training y Shape = {}\n'.format(dataset_train['SalePrice'].shape[0]))

print('Test X Shape = {}'.format(dataset_test.shape))

print('Test y Shape = {}\n'.format(dataset_test.shape[0]))
dataset_train.describe()

dataset_test.describe()
dataset_train.head(-1)
print(dataset_train.info())
print(dataset_test.info())
print(df_all.info())
drop_cols=['Alley','Fence','FireplaceQu','PoolQC','LotFrontage','MiscFeature' ]

df_all.drop(columns=drop_cols, inplace=True)


zero_cols = [

    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',

    'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'

]

for col in zero_cols:

    df_all[col].replace(np.nan, 0, inplace=True)



df_all['MSZoning'] = df_all.groupby('MSSubClass')['MSZoning'].apply(

    lambda x: x.fillna(x.mode()[0]))
conditn_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

df_all['ExterQual'] = df_all['ExterQual'].map(conditn_map).astype('int')

df_all['ExterCond'] = df_all['ExterCond'].map(conditn_map).astype('int')
def srt_reg(y, df):

    fig, axes = plt.subplots(12, 3, figsize=(25, 80))

    axes = axes.flatten()



    for i, j in zip(df.select_dtypes(include=['number']).columns, axes):



        sns.regplot(x=i,

                    y=y,

                    data=df,

                    ax=j,

                    order=3,

                    ci=None,

                    color='#3ce7e1',

                    line_kws={'color': 'green'},

                    scatter_kws={'alpha':0.4})

        j.tick_params(labelrotation=45)

        j.yaxis.set_major_locator(MaxNLocator(nbins=10))



        plt.tight_layout()
srt_reg('SalePrice', dataset_train)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor


#get correlations of each features in dataset

corrmat = df_all.corr()



top_corr_features = corrmat.index

plt.figure(figsize=(75,75))

#plot heat map

g=sns.heatmap(df_all[top_corr_features].corr(),annot=True,cmap="RdYlGn")
corrmat
corrmat['SalePrice'].sort_values(ascending=False).head(20)
df_train,df_test=divide_df(df_all)
df_train.head()
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

X=df_train[features]

y=df_train.SalePrice







# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rf_model = RandomForestRegressor()

rf_model.fit(X, y)
test_data_path = '../input/home-data-for-ml-course/test.csv'

test_data = pd.read_csv(test_data_path)

test_X = test_data[features]

X_test = test_X.fillna(train_X.mean())

test_preds = rf_model.predict(X_test )

print(test_preds)



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)