#!pip install PyForest

#from pyforest import *

#ELSE import required libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv('../input/indian-housing-price/houseprice.csv')
df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isnull().sum()/df.shape[1]
plt.subplots(figsize=(6,6))

plt.plot()

sns.boxplot(df.Age, orient='h')
sns.distplot(df.Age)
data=df.drop(['Age','Price'], axis=1)
sns.boxplot(data=data, orient='v')
sns.distplot(df['Lot Size'])
sns.distplot(df['Living Area'])
from sklearn.preprocessing import FunctionTransformer

ft=FunctionTransformer(np.log)

df['Age_log']=ft.fit_transform(df[['Age']])
ff=FunctionTransformer(np.log)

df['LotSize_log']=ft.fit_transform(df[['Lot Size']])
ff=FunctionTransformer(np.log)

df['Living_log']=ft.fit_transform(df[['Living Area']])
df.columns
plt.subplots(figsize=(6,6))

sns.boxplot(df.Age_log, orient='h')
plt.subplots(figsize=(6,6))

sns.boxplot(df.LotSize_log, orient='h')
plt.subplots(figsize=(6,6))

sns.boxplot(df.Living_log, orient='h')
df.replace(-np.inf, np.nan, inplace=True)

df.dropna(axis=0, inplace=True)
sns.distplot(df.Age_log)
sns.distplot(df.LotSize_log)
sns.distplot(df.Living_log)
sns.jointplot(df.Age_log, df.Price)
data=df.drop(['Age','Lot Size','Living Area'], axis=1)

g=sns.pairplot(data=data)
sns.countplot(df.Bedrooms)
sns.violinplot(df.Fireplace, df.Price)
sns.barplot(df.Bedrooms, df.Price, hue=df.Fireplace, palette='autumn')
sns.swarmplot(df.Bathrooms, df.Price)
df.head()
df.shape
df.isnull().sum()
df.drop(['Age','Lot Size','Living Area'], axis=1, inplace= True)
df.head()
x=df.drop(['Price'], axis=1)

y=df['Price']
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.3, random_state=1 )
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)

print("coefficients are ",lr.coef_,"intercept is ", lr.intercept_)

print()

print('r2 score for training data: ',r2_score(y_train, lr.predict(X_train)))

print('r2 score for testing data: ',r2_score(y_test, lr.predict(X_test)))

print('rmse score: ',np.sqrt(mean_squared_error(y_train, lr.predict(X_train))))
import statsmodels.api as sm

xc=sm.add_constant(x)

model=sm.OLS(y,xc).fit()

model.summary()
import statsmodels.tsa.api as smt

pattern=smt.graphics.plot_acf(model.resid, lags=40)

pattern.show()
from scipy.stats import norm

sns.distplot(model.resid, fit=norm)
import scipy.stats as st

st.jarque_bera(model.resid)
x=df.drop(['Price'], axis=1)

x=x.transform(lambda x: x**2)

y=df['Price'].transform(lambda x: x**(1/3))
import statsmodels.api as sm

xc=sm.add_constant(x)

model=sm.OLS(y,xc).fit()

model.summary()
st.jarque_bera(model.resid)
sns.regplot(x=y, y=model.predict(), lowess=True, line_kws={'color':'red'})

plt.xlabel('y actual')

plt.ylabel('y prediction')
st.probplot(model.resid, plot=plt)

plt.show()
sm.stats.linear_rainbow(res=model)
sns.regplot(x=model.predict(), y=model.resid, lowess=True, line_kws={'color':'red'})
import statsmodels.stats.api as stt

stt.het_goldfeldquandt(model.resid, model.model.exog)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]

v=pd.DataFrame({'vif':vif[:]}, index=x.columns)
v.T
sns.heatmap(df.corr(), annot=True)
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

no_of_features=np.arange(1,7)

highscore=0

score_list=[]

x=df.drop(['Price'], axis=1)

x=x.transform(lambda x: x**2)

y=df['Price'].transform(lambda x: x**(1/3))

for i in range(len(no_of_features)):

    X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=.2, random_state=1)

    model=LinearRegression()

    rfe=RFE(model, no_of_features[i])

    X_rfe_train=rfe.fit_transform(X_train,y_train)

    X_rfe_test=rfe.transform(X_test)

    model.fit(X_rfe_train,y_train)

    score=model.score(X_rfe_test,y_test)

    print(score, end=' ')

    score_list.append(score)

    if score>highscore:

        highscore=score

        nof=no_of_features[i]

print()        

print("Optimum number of feature to be selected is ", nof, " and its r2 is ",highscore)
thres=5.0

op=pd.DataFrame()

x=df.drop(['Price'], axis=1)

x=x.transform(lambda x: x**2)

y=df['Price'].transform(lambda x: x**(1/3))

k=len(x.columns)

vif=[variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

for j in range(1,k+1):

    print('iteration num ',j)

    print(vif)

    a=np.argmax(vif)

    print("the variable number is", a)

    if vif[a]<=thres:

        break

    elif j==1:

        op=x.drop(x.columns[a], axis=1)

        vif=[variance_inflation_factor(op.values,i) for i in range(op.shape[1])]

    elif j>1:

        op=op.drop(op.columns[a], axis=1)

        vif=[variance_inflation_factor(op.values,i) for i in range(op.shape[1])]       

op
x=df.drop(['Price','Bedrooms','Bathrooms'], axis=1)

x=x.transform(lambda x: x**2)

y=df['Price'].transform(lambda x: x**(1/3))

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.2, random_state=1 )

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)

print("coefficients are ",lr.coef_,"intercept is ", lr.intercept_)

print()

print('r2 score of train is: ',r2_score(y_train, lr.predict(X_train)))

print('r2 score of test is: ',r2_score(y_test, lr.predict(X_test)))

print('rmse score: ',np.sqrt(mean_squared_error(y_train, lr.predict(X_train))))
x=df.drop(['Price'], axis=1)

x=x.transform(lambda x: x**2)

y=df['Price'].transform(lambda x: x**(1/3))





import statsmodels.api as sm

cols=list(x.columns)

pmax=1

while(len(cols)>0):

    p=[]

    x_1=x[cols]

    x_1=sm.add_constant(x_1)

    model=sm.OLS(y,x_1).fit()

    p=pd.Series(model.pvalues.values[1:], index=cols)

    pmax=max(p)

    feature_of_pmax=p.idxmax()

    if (pmax>.05):

        cols.remove(feature_of_pmax)

    else:

        break;

selected_feature_BE=cols

print(selected_feature_BE)
####As per backward elimination, Bedrooms and LotSize_log features are ommitted
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error

x=df.drop(['Price','LotSize_log','Bedrooms'], axis=1)

x=x.transform(lambda x: x**2)

y=df['Price'].transform(lambda x: x**(1/3))

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.2, random_state=1 )

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)

print("coefficients are ",lr.coef_,"intercept is ", lr.intercept_)

print()

print('r2 score for training data: ',r2_score(y_train, lr.predict(X_train)))

print('r2 score for testing data: ',r2_score(y_test, lr.predict(X_test)))

print('rmse score: ',np.sqrt(mean_squared_error(y_train, lr.predict(X_train))))
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge,RidgeCV, Lasso,LassoCV

x=df.drop(['Price','Bedrooms','LotSize_log'], axis=1)

x=x.transform(lambda x: x**2)

y=df['Price'].transform(lambda x: x**(1/3))

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.2, random_state=1 )

alpha=[1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20,50,100]

ridge=Ridge()

parameters={'alpha':alpha}

ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(X_train,y_train)
r_rmse=np.sqrt(mean_squared_error(ridge_regressor.predict(X_test), y_test))

print(ridge_regressor.best_params_, "and RMSE:", r_rmse)

print("r2 score train: ",r2_score(y_train, ridge_regressor.predict(X_train)), 

                                  "r2 score test: ",r2_score(y_test, ridge_regressor.predict(X_test)))
alpha=[1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20,50,100]

lasso=Lasso()

x=df.drop(['Price','LotSize_log','Bedrooms'], axis=1)

x=x.transform(lambda x: x**2)

y=df['Price'].transform(lambda x: x**(1/3))

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.2, random_state=1 )

parameters={'alpha':alpha}

lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
l_rmse=np.sqrt(mean_squared_error(lasso_regressor.predict(X_test), y_test))

print(lasso_regressor.best_params_, "and RMSE:", l_rmse)

print("r2 score train: ",r2_score(y_train, lasso_regressor.predict(X_train)), 

                                  "r2 score test: ",r2_score(y_test, lasso_regressor.predict(X_test)))
LR=lr.predict(X_test)

RR=ridge_regressor.predict(X_test)

LL=lasso_regressor.predict(X_test)

actual=y_test.values

FinalDF=pd.DataFrame(actual, columns=['Actual'])

FinalDF['LR prediction']=LR

FinalDF['Ridge prediction']=RR

FinalDF['Lasso Prediction']=LL
FinalDF
fig, ax = plt.subplots(2,1 , figsize=(15,10))

price_head=FinalDF.head(30)

price_head.plot(kind='bar', ax=ax[0])

price_tail=FinalDF.tail(30)

price_tail.plot(kind='bar', ax=ax[1])
r2_train=[r2_score(y_train, lr.predict(X_train)), r2_score(y_train, lasso_regressor.predict(X_train)),r2_score(y_train, lasso_regressor.predict(X_train))]

r2_test=[r2_score(y_test, lr.predict(X_test)),r2_score(y_test, lasso_regressor.predict(X_test)),r2_score(y_test, ridge_regressor.predict(X_test))]

rmse=[np.sqrt(mean_squared_error(lr.predict(X_test), y_test)),np.sqrt(mean_squared_error(lasso_regressor.predict(X_test), y_test)),np.sqrt(mean_squared_error(ridge_regressor.predict(X_test), y_test))]
FinalScores=pd.DataFrame(r2_train, columns=['r2 train'], index=['LR','Lasso','Ridge'])

FinalScores['r2 test']=r2_test

FinalScores['RMSE']=rmse
FinalScores