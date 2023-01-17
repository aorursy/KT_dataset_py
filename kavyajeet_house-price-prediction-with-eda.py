# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns

import os

from scipy import stats

from scipy.special import boxcox1p

from copy import deepcopy

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

train_data.head()
train_data.shape, test_data.shape
test_data['Id'].head()
test_ids = test_data['Id']

test_start_id = test_ids.head()[0]



target_variable = train_data["SalePrice"]
# remove the target variable from the train_set

train_features = train_data.iloc[:,0:-1]

sale_price = train_data.iloc[:,-1]
all_data = pd.concat((train_features,test_data)).reset_index(drop=True)

all_data.shape
missing_values = (all_data.isna().sum()/len(all_data)*100).sort_values(ascending=False).head(40)

columns_to_remove = missing_values[missing_values>2].index

columns_to_remove
# remove the columns with more than 2% missing values

all_data = all_data.drop(columns=columns_to_remove)

all_data.shape
# Fill the rest of the missing data values with mode of each column

all_data = all_data.fillna(all_data.mode().iloc[0])

print('Number of missing data: ',all_data.isna().sum().sum())
plt.figure(figsize=(15,10))

sns.heatmap(train_data.corr().abs())

plt.show()
train_data.corr()['SalePrice'].sort_values(ascending=False)
all_data = all_data.drop(columns=['GarageArea','1stFlrSF'])

all_data.shape
all_data.dtypes.value_counts()
int_type_columns = list(all_data.dtypes[all_data.dtypes == int].index)

df = all_data[int_type_columns]

df['target'] = target_variable

df.head()
sns.pairplot(df,height=2.5, x_vars = int_type_columns, y_vars = ['target'])
# converting some of the some numerical to categorical

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)
standard_sale_price = StandardScaler().fit_transform(train_data['SalePrice'].values.reshape(-1,1))

sns.distplot(standard_sale_price)

plt.xlabel('Sale Price')

plt.ylabel('Probability')

plt.show()
sns.scatterplot(train_data['GrLivArea'],train_data['SalePrice'])

plt.show()
GrlivArea_rows_drop = train_data.loc[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000)]['Id'].values

print('indices to be removed:',GrlivArea_rows_drop)

# Verifying the index of the row in the two datasets (training and the concatenated train and test)

all_data[all_data.index==524]
train_data[train_data.index==524]
all_data = all_data.drop(index = GrlivArea_rows_drop)

sale_price = sale_price.drop(index = GrlivArea_rows_drop)
sns.scatterplot(train_data['TotalBsmtSF'], train_data['SalePrice'])

plt.show()
TotalBsmtSF_rows_drop = train_data[train_data['TotalBsmtSF'] > 6000].index.values

all_data = all_data.drop(index=TotalBsmtSF_rows_drop)

sale_price = sale_price.drop(index = TotalBsmtSF_rows_drop)
sns.distplot(train_data['SalePrice'])

plt.show()
sale_price = np.log(sale_price)

sns.distplot(sale_price)

plt.show()
sns.distplot(all_data['GrLivArea'])

plt.show()
all_data['GrLivArea'] = np.log(all_data['GrLivArea'])

sns.distplot(all_data['GrLivArea'])

plt.show()
sns.distplot(all_data['TotalBsmtSF'],fit=stats.norm)

plt.show()
sns.distplot(boxcox1p(all_data['TotalBsmtSF'],0.5), fit=stats.norm)

plt.show()
stats.probplot(boxcox1p(all_data['TotalBsmtSF'],0.5),plot=plt)

plt.show()
try_data = deepcopy(all_data)
numerical = try_data.select_dtypes(exclude='object')

categorical = try_data.select_dtypes(include='object')
std_numerical = StandardScaler().fit_transform(numerical.iloc[:,1:])

std_numerical.shape
numerical.loc[:,1:] = std_numerical
final_dataset = pd.concat([numerical,categorical],axis=1)

final_dataset.head()
## finally get dummies for categorical



final_dataset = pd.get_dummies(final_dataset)

print(final_dataset.shape)

final_dataset.head()
# now divide the concatenated dataset into training set and test set

train_dataset = final_dataset[final_dataset['Id'] < test_start_id]

test_dataset = final_dataset[final_dataset['Id'] >= test_start_id]



print(train_dataset.shape, test_dataset.shape)
len(final_dataset)
## Normalizing the target values as well

sc2 = StandardScaler()

std_target_values = sc2.fit_transform(sale_price.values.reshape(-1,1))

sns.distplot(std_target_values)

plt.show()
%%time 

regressor = Ridge(alpha=10)

X = train_dataset.values

regressor.fit(X,std_target_values)
predictions = np.exp(sc2.inverse_transform(regressor.predict(test_dataset.values).reshape(-1,1).ravel().tolist()))

len(test_ids)
prediction = {'Id':test_ids, 'SalePrice':predictions}

submission = pd.DataFrame(prediction)



submission.to_csv('submission.csv',index=False)