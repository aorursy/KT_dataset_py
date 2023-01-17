import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

data.shape
plt.figure(figsize=(13,6))



sns.distplot(data["SalePrice"]/1000, color = 'rosybrown')

plt.xlabel("Sale Price (in 1000$)")



mean = data['SalePrice'].mean()/1000

median = data['SalePrice'].median()/1000



plt.axvline(mean, color = 'darkred', label = 'Mean')

plt.axvline(median, color = 'darkolivegreen', label = 'Median')

plt.text(165,0.0070, "{}k$".format(round(mean)), color = 'darkred', rotation = 90, size = 13)

plt.text(145,0.0075, "{}k$".format(round(median)), color = 'darkolivegreen', rotation = 90, size = 13)

plt.legend()
plt.figure(figsize=(13,6))



sns.distplot(np.log(data["SalePrice"]), color = 'orangered')

plt.xlabel("Log(Sale Price) in $")
# How much missing data do we have?

total = data.isnull().sum().sort_values(ascending=False)

percent = round((100*data.isnull().sum()/data.isnull().count()).sort_values(ascending=False),1)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', '% Missing'])

missing_data.head(20)
cat_cols = [cname for cname in data.columns if data[cname].dtype == "object"]

cat_with_nans = [cname for cname in data.columns if data[cname].dtype == "object" and data[cname].isnull().any()]

print('there are', len(cat_with_nans), 'columns with missing categorical data:', cat_with_nans)
data['AgeGarage'] = data['YrSold'] - data['GarageYrBlt']

data['AgeGarage'].value_counts()
# so that we see something in the plots

data[cat_cols] = data[cat_cols].fillna("MISS")
f, ax = plt.subplots(4, 4, figsize=(20, 15))

       

for i in range(4):

    for j in range(4):

        sns.countplot(data[cat_with_nans[j+4*i]], ax=ax[i,j], palette = "bright", 

                      order = data[cat_with_nans[j+4*i]].value_counts().index.drop("MISS").insert(0, 'MISS'))

        sns.countplot(data[cat_with_nans[j+4*i]], ax=ax[i,j], palette = "bright", 

                      order = data[cat_with_nans[j+4*i]].value_counts().index.drop("MISS").insert(0, 'MISS'))

plt.show()
num_cols = [cname for cname in data.columns if data[cname].dtype == "float64" or data[cname].dtype == "int64"]

num_with_nans = [cname for cname in num_cols if data[cname].isnull().any()]

print('there are', len(num_with_nans), 'columns with missing numerical data:', num_with_nans)
# Replace all numerical values with "Values" and all NaN values with "Miss"

data.LotFrontage.where(data.LotFrontage.isnull(), "Value", inplace=True)

data.MasVnrArea.where(data.MasVnrArea.isnull(), "Value", inplace=True)

data.GarageYrBlt.where(data.GarageYrBlt.isnull(), "Value", inplace=True)

data[num_with_nans] = data[num_with_nans].fillna("MISS")
f, ax = plt.subplots(1, 3, figsize=(24, 6))

       

for j in range(3):

    sns.countplot(data[num_with_nans[j]], ax=ax[j], palette = "bright", 

                  order = data[num_with_nans[j]].value_counts().index.drop("MISS").insert(0, 'MISS'))

plt.show()
# we load the dataset again to get back the original values (with NaNs)

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id') 
sns.swarmplot(x='PoolQC', y='SalePrice', data=data, palette='Set1')
sns.swarmplot(x=data['MiscFeature'], y=data['SalePrice'])
sns.swarmplot(x=data['Alley'], y=data['SalePrice'])
data['Fence'].fillna("No", inplace = True)

sns.violinplot(x='Fence', y='SalePrice', data=data, palette='Set1', order = ["GdPrv", "MnPrv", "GdWo", "MnWw", "No"])
data['FireplaceQu'].fillna("No", inplace = True)

sns.violinplot(x='FireplaceQu',y='SalePrice',data=data,palette='Set1', order = ["Ex", "Gd", "TA", "Fa", "Po", "No"])
f, ax = plt.subplots(1, 2, figsize=(20, 5))



data['GarageQual'].fillna("No", inplace = True)

data['GarageCond'].fillna("No", inplace = True)

sns.swarmplot(x='GarageQual',y='SalePrice',data=data, palette='Set1', 

              order = ['Ex', "Gd", 'TA', 'Fa', 'Po', 'No'], ax = ax[0])

sns.swarmplot(x='GarageCond',y='SalePrice',data=data, palette='Set1', 

              order = ['Ex', "Gd", 'TA', 'Fa', 'Po', 'No'], ax = ax[1])
sns.regplot(x=data['GarageYrBlt'], y=data['SalePrice'], color = 'red', marker = "+")
f, ax = plt.subplots(1, 2, figsize=(20, 5))

sns.regplot(x=data['MasVnrArea'], y=data['SalePrice'], ax = ax[0])

sns.swarmplot(x='MasVnrType',y='SalePrice',data=data, palette='Set1', ax = ax[1])



indicies = data.index[data['MasVnrType'] == 'None']

print(data.loc[indicies]['MasVnrArea'].value_counts())



data['MasVnrType'].fillna("None", inplace = True)

data['MasVnrArea'].fillna(0, inplace = True)
sns.regplot(x=data['LotFrontage'], y=data['SalePrice'], color = 'red', marker = "+")
from sklearn.model_selection import train_test_split

import xgboost as xgb
f, ax = plt.subplots(1, 3, figsize=(20, 5))



high_cardinality_cols = [cname for cname in data.columns if data[cname].nunique() >= 10 and 

                        data[cname].dtype == "object"]



# it doesn't make much sense to decipher the x-axis so we won't bother

for i in range(len(high_cardinality_cols)):

    sns.swarmplot(x=high_cardinality_cols[i],y='SalePrice', data = data, palette='Set1', ax = ax[i])
plt.figure(figsize=(11,6))

sns.stripplot(x = data.Neighborhood, y = data.SalePrice,

              order = np.sort(data.Neighborhood.unique()),

              jitter=0.1, alpha=0.9, palette='Set1')

 

plt.xticks(rotation=45)
sns.regplot(x=data['LotArea'], y=data['SalePrice'], color = 'red', marker = "+")
sns.regplot(x=data['GrLivArea'], y=data['SalePrice'], color = 'red', marker = "+")
sns.stripplot(x = data.Heating, y = data.SalePrice,

              order = np.sort(data.Heating.unique()),

              jitter=0.1, alpha=0.5, palette='Set1')

 

plt.xticks(rotation=45)