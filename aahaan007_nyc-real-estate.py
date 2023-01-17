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



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/nyc-property-sales/nyc-rolling-sales.csv')

df.head()
df.columns
df.isnull().sum()
df.shape
df.replace({"-":0},inplace=True)
df.head()
df.dtypes
#Dropping column as it is empty

del df['EASE-MENT']

#Dropping as it looks like an iterator

del df['Unnamed: 0']



del df['SALE DATE']





df.head()
sum(df.duplicated(df.columns))
df = df.drop_duplicates(df.columns,keep = 'last')

sum(df.duplicated(df.columns))
df.shape
len(df)
df.info()
df.columns[df.isnull().any()]
df['TAX CLASS AT TIME OF SALE'] = df['TAX CLASS AT TIME OF SALE'].astype('category')

df['TAX CLASS AT PRESENT'] = df['TAX CLASS AT PRESENT'].astype('category')

df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')

df['GROSS SQUARE FEET']= pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')

df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')

df['BOROUGH'] = df['BOROUGH'].astype('category')
df.columns[df.isnull().any()]
df.isnull().sum()
miss=df.isnull().sum()/len(df)

miss=miss[miss>0]

miss.sort_values(inplace=True)

miss
miss=pd.DataFrame(miss)

miss.columns=['count']

miss.index.names=['Name']

miss['Name']=miss.index

miss

miss
sns.barplot(x = 'Name',y = 'count',data = miss)
df['LAND SQUARE FEET'] = df['LAND SQUARE FEET'].fillna(df['LAND SQUARE FEET'].mean())

df['GROSS SQUARE FEET'] = df['GROSS SQUARE FEET'].fillna(df['GROSS SQUARE FEET'].mean())
df.isnull().sum()
test = df[df['SALE PRICE'].isna()]

data = df[~df['SALE PRICE'].isna()]

test = test.drop(columns = 'SALE PRICE')
print(test.shape)

test.head()
print(data.shape)

data.head()
plt.figure(figsize=(10,10))

sns.heatmap(data.corr(),annot=True)
corr = data.corr()

corr['SALE PRICE']
np.number
numeric_data=data.select_dtypes(include=[np.number])

numeric_data.describe()
plt.figure(figsize=(15,6))

sns.boxplot(x='SALE PRICE',data = data)

plt.title('Sale Price in USD')

plt.ticklabel_format(style='plain', axis='x')

sns.distplot(data['SALE PRICE'])
data = data[(data['SALE PRICE'] > 100000) & (data['SALE PRICE'] < 5000000)]
sns.distplot(data['SALE PRICE'])
data['SALE PRICE'].skew()
sales=np.log(data['SALE PRICE'])

print(sales.skew())

sns.distplot(sales)
plt.figure(figsize=(10,6))

sns.boxplot(x='GROSS SQUARE FEET', data=data,showfliers=False)
plt.figure(figsize=(10,6))

sns.boxplot(x='LAND SQUARE FEET', data=data,showfliers=False)
data = data[data['GROSS SQUARE FEET'] < 10000]

data = data[data['LAND SQUARE FEET'] < 10000]

plt.figure(figsize=(10,6))

sns.regplot(x='GROSS SQUARE FEET', y='SALE PRICE', data=data, fit_reg=False, scatter_kws={'alpha':0.3})
plt.figure(figsize=(10,6))

sns.regplot(x='LAND SQUARE FEET', y='SALE PRICE', data=data, fit_reg=False, scatter_kws={'alpha':0.3})
data[['TOTAL UNITS','SALE PRICE']].groupby(["TOTAL UNITS"],as_index=False).count().sort_values(by = "SALE PRICE",ascending = False)
data = data[(data['TOTAL UNITS'] > 0) & (data['TOTAL UNITS'] != 2261)] 
data.head()
plt.figure(figsize=(10,6))

sns.boxplot(x='TOTAL UNITS', y='SALE PRICE', data=data)

plt.title('Total Units vs Sale Price')

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(x='RESIDENTIAL UNITS', y='SALE PRICE', data=data)

plt.title('Residential Units vs Sale Price')

plt.show()
cat_data=data.select_dtypes(exclude=[np.number])

cat_data.describe()
pivot=data.pivot_table(index='TAX CLASS AT PRESENT', values='SALE PRICE', aggfunc=np.median)

pivot
pivot.plot(kind='bar', color='black')
pivot=data.pivot_table(index='TAX CLASS AT TIME OF SALE', values='SALE PRICE', aggfunc=np.median)

pivot.plot(kind = 'bar')

pivot
pivot=data.pivot_table(index='BUILDING CLASS CATEGORY', values='SALE PRICE', aggfunc=np.median)

pivot
pivot.plot(kind = 'bar',color = 'black')
del data['ADDRESS']

del data['APARTMENT NUMBER']
data.info()
numeric_data.columns
from scipy.stats import skew

skewed = data[numeric_data.columns].apply(lambda x: skew(x.dropna().astype(float)))

skewed = skewed[skewed > 0.75]

skewed = skewed.index

data[skewed] = np.log1p(data[skewed])
data[skewed].head()
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

scalar.fit(data[numeric_data.columns])

scalar = scalar.transform(data[numeric_data.columns])

for i, col in enumerate(numeric_data.columns):

       data[col] = scalar[:,i]

data.head()
del data['BUILDING CLASS AT PRESENT']

del data['BUILDING CLASS AT TIME OF SALE']

del data['NEIGHBORHOOD']
one_hot_features = ['BOROUGH', 'BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE']
one_hot_encoded = pd.get_dummies(data[one_hot_features])

one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)
# Replacing categorical columns with dummies

fdf = data.drop(one_hot_features,axis=1)

fdf = pd.concat([fdf, one_hot_encoded] ,axis=1)
fdf.info()
Y_fdf = fdf['SALE PRICE']

X_fdf = fdf.drop('SALE PRICE', axis=1)



X_fdf.shape , Y_fdf.shape
from sklearn.model_selection import train_test_split 

X_train ,X_test, Y_train , Y_test = train_test_split(X_fdf , Y_fdf , test_size = 0.25 , random_state =42)

X_train.shape , Y_train.shape
X_test.shape , Y_test.shape
from sklearn.metrics import mean_squared_error

def rmse(y_test,y_pred):

      return np.sqrt(mean_squared_error(y_test,y_pred))
from sklearn.linear_model import LinearRegression 

linreg = LinearRegression()

linreg.fit(X_train, Y_train)

Y_pred_lin = linreg.predict(X_test)

rmse(Y_test,Y_pred_lin)

from sklearn.linear_model import Lasso

alpha=0.00099

lasso_regr=Lasso(alpha=alpha,max_iter=50000)

lasso_regr.fit(X_train, Y_train)

Y_pred_lasso=lasso_regr.predict(X_test)

rmse(Y_test,Y_pred_lasso)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.01, normalize=True)

ridge.fit(X_train, Y_train)

Y_pred_ridge = ridge.predict(X_test)

rmse(Y_test,Y_pred_ridge)
from sklearn.ensemble import RandomForestRegressor

rf_regr = RandomForestRegressor()

rf_regr.fit(X_train, Y_train)

Y_pred_rf = rf_regr.predict(X_test)

rmse(Y_test,Y_pred_rf)