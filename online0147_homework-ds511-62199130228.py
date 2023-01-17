import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt 

%matplotlib inline







dataset = pd.read_csv('../input/housesalesprediction/kc_house_data.csv') 

dataset.head()
def adjustedR2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)
from sklearn import linear_model

from sklearn.model_selection import train_test_split

train_data,test_data = train_test_split(dataset,train_size = 0.8,random_state=3)

lr = linear_model.LinearRegression()

X_train = np.array(train_data['sqft_living'], dtype=pd.Series).reshape(-1,1)

y_train = np.array(train_data['price'], dtype=pd.Series)

lr.fit(X_train,y_train)



X_test = np.array(test_data['sqft_living'], dtype=pd.Series).reshape(-1,1)

y_test = np.array(test_data['price'], dtype=pd.Series)
plt.scatter(X_train,y_train,color = 'red') 

plt.plot(X_train,lr.predict(X_train),color = 'blue') 

plt.title('Living Space (sqft) vs Price ($) (Training Set)') 

plt.xlabel('Living Space (sqft)') 

plt.ylabel('Price ($)') 

plt.show()
plt.scatter(X_test,y_test,color = 'red') 

plt.plot(X_train,lr.predict(X_train),color = 'blue') 

plt.title('Living Space (sqft) vs Price ($) (Test Set)') 

plt.xlabel('Living Space (sqft)') 

plt.ylabel('Price ($)')

plt.show()
df_dm=dataset.copy()

df_dm.describe()
dataset.shape
dataset.dtypes
dataset[dataset.isnull().any(axis=1)==True].index
dataset = dataset.replace(to_replace=[r'^\s*$', r'[?]', r'\'\s*\'', 'N/A', 'None'],

value=np.nan, regex=True)

print(dataset)
dataset.dropna(inplace=True)

dataset
import seaborn as sns

g = sns.pairplot(dataset[['price', 'bedrooms', 'bathrooms', 

     'floors', 'view', 'condition']], palette = 'hls',size=2)

g.set(xticklabels=[]);
p = sns.jointplot(data=dataset,x='view', y='price',kind='reg',color="#ff99bb")
dataset1 = dataset.drop(['date'], axis = 1)
X = dataset1.iloc[:,[2,3,4,8,10,11,12,13,14,15]]
X.head()
y = dataset1.iloc[:,[1]]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2019)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred
y_test
import statsmodels.api as sm

X2 = sm.add_constant(X) 

est = sm.OLS(y, X2)

est2 = est.fit()

print("summary()\n",est2.summary())

print("pvalues\n",est2.pvalues)

print("tvalues\n",est2.tvalues)

print("rsquared\n",est2.rsquared)

print("rsquared_adj\n",est2.rsquared_adj)



#All the attributes 

for attr in dir(est2):

    if not attr.startswith('_'):

        print(attr)



predictions = est2.predict(X2)



print(est2.predict(X2[:2,:]))



from sklearn.metrics import r2_score

print("r2_score",r2_score(y,predictions))
regressor = regressor.predict(X_test)
import statsmodels.regression.linear_model as sm

a = 0

b = 0

a, b = X.shape

X = np.append(arr = np.ones((a, 1)).astype(int), values = X, axis = 1)

print (X.shape)
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
X_opt = X[:,[2,3,4,5,6,7,8,9,10]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()