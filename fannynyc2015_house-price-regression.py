import numpy as np 

import pandas as pd 
train_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

comb_df=pd.concat((train_df, test_df), axis=0)

train_df.shape, test_df.shape, comb_df.shape 
train_df.head()

train_df.describe()
train_df['SalePrice'].describe()
import matplotlib.pyplot as plt

import seaborn as sns
#matrix correlation 

mat_corr=train_df.corr()

f, ax=plt.subplots(figsize=(12,12))

sns.heatmap(mat_corr, vmax=0.8, square=True); 

mat_corr.sort_values(by='SalePrice', ascending=False)

#pick the top 10 high correlated variables 

top_ten=mat_corr.nlargest(10, 'SalePrice')['SalePrice'].index

top_ten_corr=np.corrcoef(train_df[top_ten].values.T)

sns.set(font_scale=1.2)

sns.heatmap(top_ten_corr, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=top_ten.values, xticklabels=top_ten.values, cmap='Blues'); 

y=train_df['SalePrice']

fig=plt.figure(figsize=(15,15))

ax1=plt.subplot2grid((2,2), (0,0))

sns.boxplot(train_df['OverallQual'],y)



ax1=plt.subplot2grid((2,2), (0,1))

sns.scatterplot(train_df['GrLivArea'], y)



ax1=plt.subplot2grid((2,2), (1,0))

sns.boxplot(train_df['GarageCars'],y)

ax1=plt.subplot2grid((2,2), (1,1))

sns.scatterplot(train_df['GarageArea'],y);

fig, ax=plt.subplots(3,2, figsize=(13,13))





ax[0,0].scatter(train_df['TotalBsmtSF'], y, color='r')

ax[0,0].set_title('TotalBsmtSF')

ax[0,1].scatter(train_df['1stFlrSF'], y, color='b')

ax[0,1].set_title('TotalBsmtSF')

ax[1,0].scatter(train_df['FullBath'], y)

ax[1,0].set_title('FullBath')

ax[1,1].scatter(train_df['TotRmsAbvGrd'], y, color='black')               

ax[1,1].set_title('TotRmsAbvGrd')

ax[2,0].scatter(train_df['YearBuilt'],y, color='orange')

ax[2,0].set_title('YearBuilt')

ax[2,1].scatter(train_df['YearRemodAdd'],y, color='purple')

ax[2,1].set_title('YearRemodAdd');
comb_df
missing_data=comb_df.isnull().sum().sort_values(ascending=False)

missing_data.head(22)
missing_data=missing_data.drop('SalePrice')

col=missing_data[(missing_data>=1)].index

col
comb_df=comb_df.drop(col[:18],1) 
col=col[18:]

for c in col: 

    comb_df[c].fillna(comb_df[c].mode()[0], inplace=True)         
comb_df.shape, train_df.shape, test_df.shape
comb_df=pd.get_dummies(comb_df)

train_df=comb_df[:1460]

test_df=comb_df[1461:]
train_df=train_df.drop('Id', 1)

train_df.head()
#logistic

#Ridge 

#Lasso

#ElasticNet 

#Xgboost

#neural nets 
from sklearn.model_selection import train_test_split

x=train_df.drop('SalePrice', axis=1)

y=train_df['SalePrice']

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0 )
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, LogisticRegression

from sklearn.metrics import r2_score

from xgboost.sklearn import XGBRegressor

#logistic Regression 

logreg=LogisticRegression()

logreg.fit(x_train, y_train)

y_pred=logreg.predict(x_test)

log_score=r2_score(y_test,y_pred)

log_score
#Ridge 

ridge=RidgeCV()

ridge.fit(x_train, y_train)

y_pred=ridge.predict(x_test)

ridge_score=r2_score(y_test, y_pred)  

ridge_score

#Lasso

lasso=LassoCV()

lasso.fit(x_train, y_train)

y_pred=lasso.predict(x_test)

lasso_score=r2_score(y_test, y_pred)  

lasso_score
#ElasticNet

elastic=ElasticNetCV()

elastic.fit(x_train, y_train)

y_pred=elastic.predict(x_test)

elastic_score=r2_score(y_test, y_pred)  

elastic_score
# XGBoost



xgb=XGBRegressor()

xgb.fit(x_train, y_train)

y_pred=xgb.predict(x_test)

xgb_score=r2_score(y_test, y_pred)  

xgb_score
plt.scatter(y_test,y_pred)

plt.show()
#neural nets 

from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier()

mlp.fit(x_train,y_train)

y_pred = mlp.predict(x_test)

mlp_score=r2_score(y_test, y_pred)

mlp_score

results = pd.DataFrame({

    'Model': ['logistic','Ridge','Lasso','ElasticNet','Xgboost','neural nets'], 



    'Score': [log_score, ridge_score, lasso_score, elastic_score, xgb_score, mlp_score]

    

})



results.sort_values(by='Score', ascending=False)

x_test=test_df.drop('Id', 1)

x_test=x_test.drop('SalePrice',1)
xgb.fit(x_train, y_train)

y_pred=xgb.predict(x_test)

subm=pd.DataFrame({

    'Id': test_df.Id,

    'SalePrice':y_pred

                  })
subm