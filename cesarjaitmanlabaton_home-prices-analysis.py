import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 

# Adjusting the displays of the dataset:
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print('Dependencies installed!')
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
train.head()
print('Columns in the dataset')
train.columns
train.info()
train.describe()
train.describe(include=['O'])
plt.figure(figsize=(20, 5))
plt.scatter(train['GrLivArea'], train['SalePrice'], color='blue')
plt.title('Displaying outliers:')
plt.xlabel('SalePrice')
plt.ylabel('GrLivArea')
plt.show()
# Getting rid of the outliers:
train = train[train['GrLivArea'] < 4000]
train['SalePrice'].describe()
plt.figure(figsize=(20, 5))
sns.set(font_scale=1)

displot = sns.distplot(train['SalePrice'])
plt.show()
# Saving SalePrice in y for future training:
y = train['SalePrice']
total = train.isnull().sum().sort_values(ascending=False)
percent = ((train.isnull().sum() / train.isnull().count()) * 100).sort_values(ascending=False)
missing_values = pd.DataFrame({'Total': total, 'Missing ratio': percent})
missing_values.head(20)
plt.figure(figsize=(20, 10))

heatmap_data = train.corr()
heatmap = sns.heatmap(heatmap_data, vmax=0.9, center=0, square=True)
plt.show()
cols = ['SalePrice','OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars', 'GarageArea', 'YearBuilt', 'FullBath']
sns.set()
sns.pairplot(train[cols], size=2.5)
plt.show()
fig = plt.figure(figsize=(20, 7))

plt.subplot(121)
ax1 = plt.scatter(x='TotalBsmtSF', y='SalePrice', data=train)

plt.subplot(122)
ax2 = plt.scatter(x='1stFlrSF', y='SalePrice', data=train)

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(20, 7))

plt.subplot(121)
ax1 = plt.scatter(x='GarageCars', y='SalePrice', data=train)

plt.subplot(122)
ax2 = plt.scatter(x='GarageArea', y='SalePrice', data=train)

plt.tight_layout()
plt.show()
clean_train = train.drop((missing_values[missing_values['Total'] > 1]).index, 1)
clean_train = clean_train.drop(clean_train.loc[clean_train['Electrical'].isnull()].index)
clean_train.isnull().sum()
scaled_saleprice = StandardScaler().fit_transform(clean_train['SalePrice'][:, np.newaxis]);
print(scaled_saleprice)
plt.figure(figsize=(20, 7))

scatter_data = plt.scatter(x='GrLivArea', y='SalePrice', data=clean_train)
plt.show()
clean_train.sort_values(by='GrLivArea', ascending=False)
clean_train = clean_train.drop(clean_train[clean_train['Id'] == 1299].index)
clean_train = clean_train.drop(clean_train[clean_train['Id'] == 524].index)

plt.figure(figsize=(20, 7))

scatter_data = plt.scatter(x='GrLivArea', y='SalePrice', data=clean_train)
plt.show()
plt.figure(figsize=(20, 7))

plt.subplot(121)
sns.distplot(clean_train['SalePrice'], fit=norm);

plt.subplot(122)
res = stats.probplot(clean_train['SalePrice'], plot=plt)

plt.show()
clean_train['SalePrice'] = np.log(clean_train['SalePrice'])

plt.figure(figsize=(20, 7))

plt.subplot(121)
sns.distplot(clean_train['SalePrice'], fit=norm);

plt.subplot(122)
res = stats.probplot(clean_train['SalePrice'], plot=plt)

plt.show()
plt.figure(figsize=(20, 7))

plt.subplot(121)
sns.distplot(clean_train['GrLivArea'], fit=norm);

plt.subplot(122)
res = stats.probplot(clean_train['GrLivArea'], plot=plt)

plt.show()
clean_train['GrLivArea'] = np.log(clean_train['GrLivArea'])

plt.figure(figsize=(20, 7))

plt.subplot(121)
sns.distplot(clean_train['GrLivArea'], fit=norm);

plt.subplot(122)
res = stats.probplot(clean_train['GrLivArea'], plot=plt)

plt.show()
