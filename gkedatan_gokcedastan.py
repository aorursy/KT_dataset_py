# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from scipy.stats import norm, skew
pd.options.mode.chained_assignment = None  # default='warn'

import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
## 1 -missing value olan özelliklerin listesi

features_with_na=[features for features in train.columns if train[features].isnull().sum()>1]
for feature in features_with_na:
    print(feature, np.round(train[feature].isnull().mean(), 4),  ' % missing values')
#plot of missing value attributes
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull())
plt.show()

train.info()

train.describe()
test.describe()
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show() 
train['SalePrice'] = np.log1p(train['SalePrice'])

#Check again for more normal distribution

plt.subplots(figsize=(12,9))
sns.distplot(train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
plt.show()
#korelasyon matrisini grafikleştiriyoruz. Id colonuna gerek olmadığı için onu çıkarıyoruz
train_corr = train.select_dtypes(include=[np.number])
del train_corr['Id']

#Coralation plot
corr = train_corr.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)

##ev fiyatlarını etkileyen değişkenleri gösteriyoruz
top_feature = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
#missing value
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max()
train.isnull().sum().sort_values(ascending=False).head(7)
#missing value
total = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
test = test.drop((missing_data[missing_data['Total'] > 1]).index,1)
test = test.drop(train.loc[train['Electrical'].isnull()].index)
test.isnull().sum().max()
test.isnull().sum().sort_values(ascending=False)

#PREPROCESSING
train = train[train.SalePrice<450000]
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] =16.0
fig_size[1] = 4.0

x =train['SalePrice']
plt.hist(x, density=True, bins=400)
plt.ylabel('SalePrice');
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) ].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#saleprice correlation matrix
k = 15 #number of variables for heatmap
plt.figure(figsize=(12,7))
corrmat = train.corr()
# picking the top 15 correlated features
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
train = train[cols]
test= test[cols.drop(['SalePrice'])]
test

#burada train ve test datalarına ayırıyoruz. bir kısmı eğitmek ve bir kısmı test etmek için.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice', axis=1), train['SalePrice'], test_size=0.2)
#Linear REgression

#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()
#Fit the model
# we are going to scale to data

y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)
# fittransform ile float değerlere çeviriyoruz ki model çalışabilsin. 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_X.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)
model.fit(X_train,y_train)
print(model)
#model oluşturuldu
predictions = model.predict(X_test)
predictions= predictions.reshape(-1,1)
plt.figure(figsize=(10,4))
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
plt.figure(figsize=(10,4))
plt.plot(y_test,label ='Test')
plt.plot(predictions, label = 'predict')
plt.show()
from sklearn import metrics

#bir modelin doğru çalışıp çalışmadıpını ölçmek için kayıp fonk. kullanıyoruz. 0'a ne kadar yakın olursa o kadar iyi çalışır.
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Accuracy: ", model.score(X_test, y_test)*100)

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
cols=cols.drop(['SalePrice']) 

for item in cols:
    print(item)
    if (is_string_dtype(test[item])):
        test[item]=test[item].fillna(mostCommon(item))
    elif (is_numeric_dtype(test[item])):
        test[item]=test[item].fillna(test[item].mean())
test_id = test_df['Id']
test_df = pd.DataFrame(test_id, columns=['Id'])
test_df

test = sc_X.fit_transform(test)

test
test_prediction=model.predict(test)
test_prediction= test_prediction.reshape(-1,1)
test_prediction =sc_y.inverse_transform(test_prediction)

sc_y.inverse_transform(test_prediction)

test_prediction = pd.DataFrame(test_prediction, columns=['SalePrice'])

test_prediction.head()

result = pd.concat([test_df,test_prediction], axis=1)

result.head()

result.to_csv('submission.csv',index=False)

