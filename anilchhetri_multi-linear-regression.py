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
import pandas as pd

import matplotlib as mpl 

import matplotlib.pyplot as plt



mpl.style.use('seaborn-darkgrid')
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.info()
df.describe(include=['O'])
df.describe()
import seaborn as sns
fig, ax = plt.subplots(figsize=(15,8))

sns.set_context('talk')

sns.boxplot(data=df.drop(columns='salary'),ax=ax)
import numpy as np

fig, ax = plt.subplots(figsize=(5,10))

_= df['hsc_p'].plot(kind='box', ax=ax,)

plt.axhline(np.mean(df['hsc_p']), xmin= 0.25, xmax=0.75, label='Mean', color='red')

plt.axhline(np.median(df['hsc_p']), xmin= 0.3, xmax=0.7, label='Median', color='blue')

ax.legend()



### mean is greater than median hence Right skewed data
#removing outliears



#calculating upper whiskers



hsc_p = df['hsc_p'].describe()

UWhiskers = hsc_p['75%'] + 1.5 * (hsc_p['75%'] - hsc_p['25%'])

LWhiskers = hsc_p['25%'] - 1.5 * (hsc_p['75%'] - hsc_p['25%'])

print(f'Upper Whiskers : {UWhiskers}')

print(f'Upper Whiskers : {LWhiskers}')
df['hsc_p'].loc[df['hsc_p'] > UWhiskers] = UWhiskers

df['hsc_p'].loc[df['hsc_p'] < LWhiskers] = LWhiskers
fig, ax = plt.subplots(figsize=(5,10))

_= df['hsc_p'].plot(kind='box', ax=ax,)

plt.axhline(np.mean(df['hsc_p']), xmin= 0.25, xmax=0.75, label='Mean', color='red')

plt.axhline(np.median(df['hsc_p']), xmin= 0.3, xmax=0.7, label='Median', color='blue')

ax.legend()
#Checking distribution of Target variables



df['mba_p'].describe()

#sscp & hscp

X = df[['ssc_p', 'hsc_p']]

y = df[['mba_p']]

print(X.head())

print(y.head())
from sklearn.model_selection import train_test_split



xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=123)
print(f'shape of train data : {xtrain.shape} : {ytrain.shape}')

print(f'shape of train data : {xtest.shape} : {ytest.shape}')
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()

mlr.fit(xtrain, ytrain)
mlr.intercept_, mlr.coef_
yhat = mlr.predict(X)
fig, ax = plt.subplots(figsize=(10,10))

sns.distplot(ytrain, label='Actual Value', hist=False, ax=ax)

sns.distplot(mlr.predict(xtrain), label='Predicted Value', hist=False, ax=ax)

ax.legend()

_ = ax.set_title('Actual Value Vs Predicted Value for Train Data')
# from Above figure we can see that it didn't do well when features values were ssc_p, hsc_p

print(f'the score = {mlr.score(X, y)* 100}')
sns.regplot(X.iloc[:, 0], y)
sns.regplot(X.iloc[:, 1], y)
pd.concat([X,y], axis=1).corr()
fig, ax = plt.subplots(figsize=(10,10))

sns.distplot(ytest, label='Actual Value', hist=False, ax=ax)

sns.distplot(mlr.predict(xtest), label='Predicted Value', hist=False, ax=ax)

ax.legend()

_ = ax.set_title('Actual Value Vs Predicted Value for Test Data')
from sklearn.metrics import mean_squared_error
print(f'the mean square error train data is {mean_squared_error(ytrain, mlr.predict(xtrain))}')

print(f'the mean square error test data is {mean_squared_error(ytest, mlr.predict(xtest))}')
mlr2 = LinearRegression()

x = df[['ssc_p', 'degree_p']]

y = df[['mba_p']]

xtrain, xtest, ytrain, ytest = train_test_split(x , y, test_size=0.2, random_state=0)

mlr2.fit(xtrain, ytrain)
fig,ax=plt.subplots(1,2, figsize=(20,10))





sns.distplot(ytrain, hist=False, color='green', label='Actual', ax=ax[0])

sns.distplot(mlr2.predict(xtrain), hist=False, color='red', label='predicted', ax=ax[0])

ax[0].legend()

ax[0].set_title('Actual vs predicted Train Data')





sns.distplot(ytest, hist=False, color='green', label='Actual', ax=ax[1])

sns.distplot(mlr2.predict(xtest), hist=False, color='red', label='predicted', ax=ax[1])

ax[1].legend()

ax[1].set_title('Actual vs predicted Test Data')
print('The Score of model (test Data )= ' ,mlr2.score(xtest,ytest) *100)

print('The Score of model (Train Data )= ' ,mlr2.score(xtrain,ytrain) *100)



print(f'The Regression Line is given by : \n y = {round(mlr2.coef_[0,0], 4)} * {x.columns.values[0]} + {round(mlr2.coef_[0,1], 4)} * {x.columns.values[1]}  + {mlr2.intercept_[0]} ' )
pd.concat([x,y], axis='columns').corr()
fig, ax = plt.subplots(2,2, figsize=(20,20), sharey=True, sharex=True)



sns.residplot(xtrain.iloc[:,0], ytrain, ax=ax[0,0])

sns.residplot(xtrain.iloc[:,1], ytrain, ax=ax[0,1])



sns.residplot(xtest.iloc[:,0], ytest, ax=ax[1,0])

sns.residplot(xtest.iloc[:,1], ytest, ax=ax[1,1])
