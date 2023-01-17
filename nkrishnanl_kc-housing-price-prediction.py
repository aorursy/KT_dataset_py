# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score as acc

from sklearn.metrics import r2_score

from sklearn import ensemble
df = pd.read_csv("../input/kc-housesales-data/kc_house_data.csv")
df.head(5)
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()
df.bedrooms.value_counts()
sns.countplot(df.bedrooms)
df.floors.value_counts()
sns.countplot(df.floors)
df.grade.value_counts()
sns.countplot(df.grade)
sns.boxplot(df.grade,df.price)
sns.regplot(df.grade,df.price)
sns.distplot(df.price)
df['price'] = df['price'].apply(lambda x : np.log(x))
sns.distplot(df.price)
continous_columns=['id','price', 'bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'waterfront','sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

       'lat', 'long', 'sqft_living15', 'sqft_lot15']

for i in continous_columns:

    print("columns name:",i)

    sns.distplot(df[i])

    plt.show()
for i in continous_columns:

    print("column name:",i)

    sns.boxplot(df[i])

    plt.show()
mod=[]

def find_outlier(df_in, col_name):

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    df_high=df_in.loc[(df_in[col_name] > fence_high)]

    #df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

    outlier_percentage=(df_high.shape[0]/len(df))*100

    outlier_percentage=round(outlier_percentage,2)

    print(outlier_percentage)

    mod.append((col_name,outlier_percentage))
mod
for i in continous_columns:

    find_outlier(df,i)
length=len(continous_columns)
result=[]

names=[]

for i in range(0,length-1):

    result.append(mod[i][1])

    names.append(mod[i][0])
    names
p=pd.DataFrame(result)

p.rename(columns={0:"outlier_percentage"},inplace=True)
p.plot.bar(figsize=(20, 8))

plt.xticks(np.arange(length),names)

plt.title("percentage of outliers in each column")

plt.axhline(5)#axis 

plt.show()
df.drop(['date'],axis=1,inplace=True)
df.corr()
df.cov()
fig,ax= plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.heatmap(df.corr(),annot=True,linewidths=0.2,vmax = .9)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
y=df['price']

X=df.drop('price',axis=1)

xc=sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
import scipy.stats as stats

print(np.abs(round(stats.norm.isf(q = 0.025),2)))
lin_reg = LinearRegression()

model = lin_reg.fit(X_train,y_train)

print(f'R^2 score for train: {lin_reg.score(X_train, y_train)}')

print(f'\nR^2 score for test: {lin_reg.score(X_test, y_test)}')
from scipy.stats import ttest_1samp

print(df.mean(),np.std(df,ddof = 1))
Xc = sm.add_constant(X)

lin_reg = sm.OLS(y,Xc).fit()

lin_reg.summary()
lin_reg = LinearRegression()

lin_reg.fit(X, y)



print(f'Coefficients: {lin_reg.coef_}')

print(f'\nIntercept: {lin_reg.intercept_}')

print(f'\nR^2 score: {lin_reg.score(X, y)}')
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

model = ensemble.GradientBoostingRegressor(**params)



model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score

model_score = model.score(X_train,y_train)

# Have a look at R sq to give an idea of the fit ,

# Explained variance score: 1 is perfect prediction

print('R2 sq: ',model_score)

y_predicted = model.predict(X_test)



# The mean squared error

print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predicted))

# Explained variance score: 1 is perfect prediction

print('Test Variance score: %.2f' % r2_score(y_test, y_predicted))

print("RMSE:%.2f"% np.sqrt(mean_squared_error(y_test, y_predicted)))
print(f'R^2 score for train: {model.score(X_train, y_train)}')

print(f'\nR^2 score for test: {model.score(X_test, y_test)}')
from sklearn.model_selection import cross_val_predict



fig, ax = plt.subplots()

ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Actual')

ax.set_ylabel('Predicted')

ax.set_title("Actual vs Predicted")

plt.show()