import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
mtcars=pd.read_csv('../input/linear-regression-eda-python/mtcars.csv')

mtcars.head()
mtcar1=mtcars

mtcar1.head()
mtcar1.info()
mtcar1.nunique()
mtcar1.cyl.unique()
mtcar1.vs.unique()
mtcar1.am.unique()
mtcar1.gear.unique()
mtcar1.carb.unique()
mtcar1.describe()
plt.boxplot(mtcar1.mpg)

plt.show()
plt.boxplot(mtcar1.disp)

plt.show()
plt.boxplot(mtcar1.hp)

plt.show()
plt.boxplot(mtcar1.drat)

plt.show()
plt.boxplot(mtcar1.wt)

plt.show()
plt.boxplot(mtcar1.qsec)

plt.show()
f,ax=plt.subplots(1,2,figsize=(10,5))

mtcar1.vs.value_counts().plot(kind='bar',ax=ax[0])

mtcar1.am.value_counts().plot(kind='bar',ax=ax[1])

plt.show()
f,ax=plt.subplots(1,2,figsize=(10,5))

mtcar1.cyl.value_counts().plot(kind='bar',ax=ax[0])

mtcar1.gear.value_counts().plot(kind='bar',ax=ax[1])

plt.show()
sns.countplot(mtcar1.gear,hue=mtcar1.cyl)

plt.show()
mtcar2=mtcar1.drop(['cyl','vs','am','gear','carb'],axis=1)

mtcar2.head()
mtcar1.groupby(['cyl'])['carb'].value_counts()[8].plot(kind='bar')

plt.show()
mtcar1.groupby(['vs','am'])['gear'].value_counts()[1][1].plot(kind='bar')

plt.show()
mtcar1.groupby(['vs','am'])['gear'].value_counts()[0][1].plot(kind='bar')

plt.show()
mtcar1.groupby(['vs','am'])['carb'].value_counts()[1][1].plot(kind='bar')

plt.show()
mtcar1.groupby(['vs','am'])['carb'].value_counts()[1][0].plot(kind='bar')

plt.show()
sns.pairplot(mtcar2)

plt.show()
X=mtcar1.drop(['mpg','model'],axis=1)

Y=mtcar1.mpg

import statsmodels.api as sm

X_constant = sm.add_constant(X)

model = sm.OLS(Y,X_constant).fit()

model.summary()

from sklearn.linear_model import LinearRegression

lin_reg1=LinearRegression()

lin_reg1.fit(X,Y)

lin_reg1.score(X,Y)
import statsmodels.tsa.api as smt



acf = smt.graphics.plot_acf(model.resid, alpha=0.05) # ACF is auto correlation function

acf.show()
from scipy.stats import jarque_bera

name=['ch-stat','p-value']

values=jarque_bera(model.resid)

from statsmodels.compat import lzip

jb=lzip(name,values)

print(jb)
sns.distplot(model.resid)

plt.show()
mean_res=model.resid.mean()

print('Mean of residuals is %.6f'%mean_res)
y_pre=model.predict(X_constant)

f,ax=plt.subplots(1,2,figsize=(10,8))

sns.regplot(Y,y_pre,ax=ax[0])

sns.regplot(model.resid,y_pre,ax=ax[1])

plt.show()
test = sm.stats.diagnostic.linear_rainbow(res=model)

print(test)
name = ['F statistic', 'p-value']

import statsmodels.stats.api as sms

test = sms.het_goldfeldquandt(model.resid, model.model.exog)

lzip(name, test)
sns.set_style('whitegrid')

sns.residplot(y_pre,model.resid,lowess=True,color = 'g')

plt.xlabel('Predicted')

plt.ylabel('Residual')

plt.title('Residual vs Predicted')

plt.show()
sns.heatmap(mtcar1.corr(),annot=True)

plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

col=X_constant.shape[1]

vif=[variance_inflation_factor( X_constant.values,i) for i in range(col)]

vif_pd=pd.DataFrame({'vif':vif[1:]},index=X.columns).T

vif_pd
X1=mtcar1.drop(['cyl','disp','hp','wt','qsec','gear','model','mpg','carb'],axis=1)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X1,Y,test_size=0.3,random_state=1)

X1.head()
X_constant1 = sm.add_constant(x_train)

model1 = sm.OLS(y_train,X_constant1).fit()

model1.summary()

from sklearn import metrics

x_cont_train = sm.add_constant(x_train)

x_cont_test = sm.add_constant(x_test)

y_tr_pred=model1.predict(x_cont_train)

y_tst_pred=model1.predict(x_cont_test)

print('R2 for train:',metrics.r2_score(y_train,y_tr_pred))

print('R2 for test:',metrics.r2_score(y_test,y_tst_pred))
lin_reg2=LinearRegression()

lin_reg2.fit(x_train,y_train)

print('R2 for train:',lin_reg2.score(x_train,y_train))

print('R2 for test:',lin_reg2.score(x_test,y_test))
q1=mtcar1.mpg.quantile(0.25)

q3=mtcar1.mpg.quantile(0.75)

iqr=q3-q1

ll=q1-1.5*iqr

ul=q3+1.5*iqr

print(mtcar1.shape)

mtcar3 = mtcar1[~((mtcar1['mpg']<ll) | (mtcar1['mpg']>ul))]

print(mtcar3.shape)

X_wo=mtcar3.drop(['mpg','model'],axis=1)

Y_wo=mtcar3['mpg'].values

X_const_wo=sm.add_constant(X_wo)

model_wo=sm.OLS(Y_wo,X_const_wo).fit()

model_wo.summary()
name = ['F statistic', 'p-value']

import statsmodels.stats.api as sms

test_wo = sms.het_goldfeldquandt(model_wo.resid, model_wo.model.exog)

lzip(name, test_wo)

vif1=[variance_inflation_factor( X_const_wo.values,i) for i in range(X_const_wo.shape[1])]

vif_pd1=pd.DataFrame({'vif':vif1[1:]},index=X_wo.columns).T

vif_pd1

lin_reg_wo=LinearRegression()

X_wo1=X_wo[['drat','vs','am']]



x_train1,x_test1,y_train1,y_test1=train_test_split(X_wo1,Y_wo,test_size=0.3,random_state=2)

lin_reg_wo.fit(x_train1,y_train1)



print(lin_reg_wo.score(x_train1,y_train1))

print(lin_reg_wo.score(x_test1,y_test1))