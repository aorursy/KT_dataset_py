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
Train=pd.read_csv('../input/train.csv',index_col='Id')

Test=pd.read_csv('../input/test.csv',index_col='Id')

Sample=pd.read_csv('../input/sample_submission.csv',index_col='Id')
pd.set_option('display.max_columns', None)

Train.head()
# Categorical Columns



Train_category_data=Train.select_dtypes(include='object')

Train_category_column=list(Train_category_data.columns)

print(f'Name of Categorical Columns: \n {Train_category_column},\n\n Length of Categorical columns: {len(Train_category_column)}')
# Numerical_value COlumns



Train_num_cat_data=Train.select_dtypes(exclude='object')

Train_num_cat_column=list(Train_num_cat_data.columns)

print(f'Name of Numerical_Categorical Columns: \n {Train_num_cat_column},\n\n Length of Numerical_Categorical columns: {len(Train_num_cat_column)}')
CORR=Train.corr()

# Values of CORR of different features of data with respect to 'SalePrice'

CORR['SalePrice'].sort_values(ascending=False)[:20]
Higher_Corr_list=list((CORR['SalePrice'].sort_values(ascending=False)[:20]).index)

Higher_Corr_list
import matplotlib.pyplot as plt

import seaborn as sns



fig = plt.figure(figsize=(10,40))



for i in range(len(Higher_Corr_list)):

    fig.add_subplot(10, 2, i+1)

    sns.regplot(Train[Higher_Corr_list].iloc[:,i], Train['SalePrice'])

    

plt.tight_layout()

plt.show()
Train=Train.drop(Train[(Train['GrLivArea'] > 4000) & (Train['SalePrice'] <200000)].index) #1

Train=Train.drop(Train[Train['GarageArea']>1200].index) #2

Train=Train.drop(Train[Train['TotalBsmtSF']>4000].index) #3

Train=Train.drop(Train[Train['1stFlrSF']>4000].index) #4

Train=Train.drop(Train[Train['MasVnrArea']>1400].index) #5

Train=Train.drop(Train[Train['BsmtFinSF1']>4000].index) #6

Train=Train.drop(Train[Train['LotFrontage']>300].index) #7

Train=Train.drop(Train[Train['OpenPorchSF']>500].index) #7
Train.shape
# Total Missing Values including Categorical and Numerical Data

(Train.isnull().sum().sort_values(ascending=False)[:21])
# For Categorical Data Missing Values

Train[Train_category_column].isnull().sum().sort_values(ascending = False)[:17]

Null_cat_column = ['PoolQC', 'MiscFeature', 'Alley', 'Fence' , 'FireplaceQu' ,'GarageCond' , 'GarageQual','GarageFinish','GarageType', 'BsmtFinType2',

                   'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond', 'MasVnrType', 'Electrical']
Null_cat_column = ['PoolQC', 'MiscFeature', 'Alley', 'Fence' , 'FireplaceQu' ,'GarageCond' , 'GarageQual','GarageFinish','GarageType', 'BsmtFinType2',

                   'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond', 'MasVnrType', 'Electrical']
d={}

for col in Null_cat_column:

    i= len(Train[col].unique())

    d[col]= Train[col].unique()[:i]
d
## Replacce th None

cat_None=['PoolQC', 'MiscFeature', 'Alley', 'Fence' , 'FireplaceQu' ,'GarageCond' , 'GarageQual','GarageFinish','GarageType', 'BsmtFinType2',

                   'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond']



for col in cat_None:

    Train[col]=Train[col].fillna('None').astype(str)
Train['MasVnrType']=Train['MasVnrType'].fillna(Train['MasVnrType'].mode()[0])

Train['Electrical']=Train['Electrical'].fillna(Train['Electrical'].mode()[0])
Train[['LotFrontage','LotConfig', 'LotArea']]
Train['LotFrontage']=Train['LotFrontage'].fillna(Train['LotFrontage'].mean())
print(Train[['MasVnrArea','MasVnrType']])
Train['MasVnrArea']=Train['MasVnrArea'].fillna(0)
Train[['GarageYrBlt','YearBuilt']]
index = list(Train['GarageYrBlt'].index[Train['GarageYrBlt'].apply(np.isnan)])

#(Train['GarageYrBlt'].index.isnull())
index
for i in index:

        Train['GarageYrBlt'][i]=Train['YearBuilt'][i]

#Train['GarageYrBlt']= Train['GarbageYrBlt'].map(Train['YearBuilt'])
Train['GarageYrBlt']
Y=Train['SalePrice']

Train=Train.drop(columns='SalePrice')

Train.shape
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

for col in Train_category_column:

    Train[col]=label.fit_transform(Train[col])
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
X_t,x_t,Y_t,y_t=train_test_split(Train,Y, test_size=0.3)

model=RandomForestRegressor()
model.fit(X_t,Y_t)
y_p=model.predict(x_t)
from sklearn.metrics import accuracy_score, mean_absolute_error



MAE= mean_absolute_error(y_p,y_t)

MAE