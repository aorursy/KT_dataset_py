import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df= pd.read_csv('../input/train.csv')

df.head()
df.columns
sns.distplot(df['SalePrice'])
print("Skewness: %f" % df['SalePrice'].skew())

print("mean: %f" % df['SalePrice'].mean())

print("median: %f" % df['SalePrice'].median())
#saleprice correlation matrix

#number of variables for heatmap

k = 6

corrmat = df.corr()

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
data = pd.concat([df['SalePrice'], df['OverallQual']], axis=1)

sns.boxenplot(x='OverallQual', y='SalePrice',data=data)
data = pd.concat([df['SalePrice'], df['GrLivArea']], axis=1)

sns.regplot(x='GrLivArea', y='SalePrice',data=data);
data = pd.concat([df['SalePrice'], df['GarageCars']], axis=1)

sns.boxenplot(x='GarageCars', y='SalePrice',data=data)
data = pd.concat([df['SalePrice'], df['TotalBsmtSF']], axis=1)

sns.regplot(x='TotalBsmtSF', y='SalePrice', data=data);
df_final=df[['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','SalePrice']]

df_final
#df_final_2=df[['LotArea','OverallQual','OverallCond','TotalBsmtSF','FullBath','HalfBath','BedroomAbvGr','TotRmsAbvGrd','Fireplaces','GarageArea','SalePrice']]

missing_data = df_final.isnull().sum().sort_values(ascending=False)

missing_data
dataset=df_final.values

dataset
X=df_final.iloc[:,0:4]

y=dataset[:,4]

y
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

X_scale = min_max_scaler.fit_transform(X)

X_scale
test=pd.read_csv('../input/test.csv')

test.head()
test_final=test[['OverallQual','GrLivArea','GarageCars','TotalBsmtSF']]

test_final.head()
test_scale = min_max_scaler.fit_transform(test_final)

test_scale
print(X_scale.shape, y.shape,test_final.shape)
from xgboost import XGBRegressor



my_model = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(X_scale, y, verbose=False)
predictions = my_model.predict(test_scale)

my_model.score(X_scale,y)
my_submission = pd.DataFrame({'Id':test.Id, 'SalePrice': predictions})

my_submission.to_csv('submission.csv',index = False)