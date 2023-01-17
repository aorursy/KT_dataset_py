import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
housingprice = pd.read_csv('../input/train.csv')
def show_missing():

    missing = housingprice.columns[housingprice.isnull().any()].tolist()

    return missing
housingprice[show_missing()].isnull().sum()
housingprice.describe()
housingprice.info()
housingprice['LotArea'].corr(housingprice['LotFrontage'])
housingprice['sqrtarea'] = np.sqrt(housingprice['LotArea'])
housingprice['sqrtarea'].corr(housingprice['LotFrontage'])

                              
sns.pairplot(housingprice[['sqrtarea','LotFrontage']].dropna())
mi = housingprice['LotFrontage'].isnull()

housingprice.LotFrontage[mi] = housingprice.sqrtarea[mi]
del housingprice['sqrtarea']
housingprice['Alley'].value_counts()
housingprice.loc[housingprice['MasVnrType'].isnull(),'MasVnrType'] = 'none'
housingprice.loc[housingprice['MasVnrArea'].isnull(),'MasVnrArea'] = 0
def cat_imputation(column, value):

    housingprice.loc[housingprice[column].isnull(),column] = value
housingprice['Fireplaces'][housingprice['FireplaceQu'].isnull()==True].describe()
garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

housingprice[garage_cols][housingprice['GarageType'].isnull()==True]
for cols in garage_cols:

    if housingprice[cols].dtype==np.object:

        cat_imputation(cols,'None')

    else:

        cat_imputation(cols, 0)
housingprice['PoolArea'][housingprice['PoolQC'].isnull()==True].describe()
cat_imputation('PoolQC', 'None')
cat_imputation('Fence', 'None')
cat_imputation('MiscFeature', 'None')