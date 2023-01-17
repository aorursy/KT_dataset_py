from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

import warnings 

warnings.filterwarnings('ignore')

import statsmodels.api as sm

import statsmodels.tsa.api as smt

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

from statsmodels.compat import lzip

from statsmodels.compat import lzip

import statsmodels.stats.api as sms

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score as acc

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

import matplotlib

df=pd.read_csv('../input/insurance-premium-prediction/insurance.csv')
df.head()
df.shape
df.dtypes
df.describe()
df.describe(include='object')
df.isnull().sum()
#outliers for age



ul=51+1.5*(51-27)

ll=51-1.5*(51-27)

print(ll,ul)
#listing outliers

print(df[df['age']<15])

print(df[df['age']>87])
#outliers for charges

ul=16639+1.5*(16639-4740)

ll=16639-1.5*(1663-4740)

print(ll,ul)
#listing outliers

print(df[df['expenses']<21254])

print(df[df['expenses']>34487])
#outliers for bmi

ul=34.69+1.5*(34.69-26.29)

ll=34.69-1.5*(34.69-26.29)

print(ll,ul)
#listing outliers

print(df[df['bmi']<22.09])
#listing outliers

print(df[df['bmi']>47.289])
#a

# Treating outliers

df['expenses']=np.log(df['expenses'])

df['expenses'].plot(kind='box')
df['bmi'].plot(kind='box')
df['bmi']=np.sqrt(df['bmi'])

df['bmi'].plot(kind='box')
corr=df.corr()

ax=sns.heatmap(corr,annot=True)

bottom,top = ax.get_ylim()

ax.set_ylim(bottom+0.5,top -0.5)
sns.scatterplot(x='expenses',y='age',data=df)
# converting sex,region and smoker columns

df=pd.get_dummies(data=df,columns=['sex','region','smoker'])
df.head()
df1=df.copy()
#Defining X and y to build the regression model

X=df.drop('expenses',axis=1)

y=df['expenses']
#Splitting into train and test data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)
lin_reg=LinearRegression()

lin_reg.fit(X,y)
print('Co-efficients:',lin_reg.coef_)
print('r2:',lin_reg.score(X,y))
from sklearn.metrics import mean_squared_error as mse
#rmse calculation

y_pred=lin_reg.predict(X)

mse(y,y_pred)
rmse=np.sqrt(0.002531)
print('The rmse value is:',rmse)
from sklearn.metrics import mean_absolute_error as mae
mae(y,y_pred)
#ols model

X_constant = sm.add_constant(X)

lin_reg=sm.OLS(y,X_constant).fit()

lin_reg.summary()
fig_d=(10,10)

fig,ax=plt.subplots(figsize=fig_d)

corr=df.corr()

ax=sns.heatmap(corr,annot=True,ax=ax)

bottom,top = ax.get_ylim()

ax.set_ylim(bottom+0.5,top -0.5)
# selecting highly correlated features

cor_target=abs(corr['expenses'])

relevant_features=cor_target[cor_target>0.5]

relevant_features
#LASSO 

reg=LassoCV()

reg.fit(X,y)

coef=pd.Series(reg.coef_,index=X.columns)

imp_coef=coef.sort_values()

matplotlib.rcParams['figure.figsize']=(8.0,10.0)

imp_coef.plot(kind='barh')

plt.title('Important features using LASSO')
X=df1.drop(['expenses','bmi','region_southeast','region_southwest'],axis=1)

y=df1['expenses']
#ols model

X_constant = sm.add_constant(X)

lin_reg=sm.OLS(y,X_constant).fit()

lin_reg.summary()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)
lin_reg=LinearRegression()

lin_reg.fit(X,y)
print('r2:',lin_reg.score(X,y))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)
print('r2 for train:',lin_reg.score(X_train,y_train))

print('r2 for test:',lin_reg.score(X_test,y_test))