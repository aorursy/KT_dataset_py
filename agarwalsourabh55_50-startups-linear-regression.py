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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#% matplotlib inline
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
'''

'''
n_samples = 100
X = np.linspace(0, 10, 100)
y = X ** 3 + np.random.randn(n_samples) * 100 + 100
plt.figure(figsize=(10,8))
plt.scatter(X, y)
'''

'''
lin_reg = LinearRegression()
lin_reg.fit(X.reshape(-1, 1), y)
model_pred = lin_reg.predict(X.reshape(-1,1))
plt.figure(figsize=(10,8));
plt.scatter(X, y);
plt.plot(X, model_pred);
print(r2_score(y, model_pred))
'''
'''
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=1)
X_poly = poly_reg.fit_transform(X.reshape(-1, 1))
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y.reshape(-1, 1))
y_pred = lin_reg_2.predict(X_poly)
plt.figure(figsize=(10,8));
plt.scatter(X, y);
plt.plot(X, y_pred);
print(r2_score(y, y_pred))
'''
data=pd.read_csv('../input/50-startups/50_Startups.csv')
data
train=data.iloc[:,:3]
test=data.iloc[:,3]

import matplotlib.pyplot as plt 
for i in train.columns:
     plt.figure()
     plt.scatter(train[i],test)
'''
plt.figure()
for i in range(3):
    #plt.subplot(3,1)
    plt.figure()
    
    plt.scatter(train.iloc[:,i],test)
    m,b=np.polyfit(train.iloc[:,i],test,1)
    plt.plot(train.iloc[:,i], m*train.iloc[:,i] + b)
'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lr=LinearRegression()
lr.fit(train,test)
#plt.figure()
pred=lr.predict(train)
for i in train.columns:
    plt.figure()
    plt.scatter(train[i],test)
    #plt.figure()
    plt.plot(train[i],pred)
    print(r2_score(test, pred))

import statsmodels.api as sm
model=sm.OLS(test, train).fit()
model.summary()
pred=model.predict(train)
r2_score(pred,test)

#according to coef_ if  I drop 2 index then it will not have an impact on model 
train=train.drop(['Administration'],axis=1)
lr=LinearRegression()
lr.fit(train,test)
pred=lr.predict(train)
plt.figure()
plt.scatter(train.iloc[:,0],test)
#plt.figure()
plt.plot(train.iloc[:,0],pred)
print(r2_score(test, pred))

lr.coef_
lr.rank_
lr.singular_
lr.intercept_
lr=LinearRegression(normalize='l2')
lr.fit(train,test)
pred=lr.predict(train)
plt.figure()
plt.scatter(train.iloc[:,0],test)
#plt.figure()
plt.plot(train.iloc[:,0],pred)
print(r2_score(test, pred))

from sklearn.linear_model import Ridge
rr=Ridge()
rr.fit(train,test)


pr=rr.predict(train)
r2_score(pr,test)
