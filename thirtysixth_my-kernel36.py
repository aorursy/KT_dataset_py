import pandas as pd

import numpy as np

data=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

x=data.drop('SalePrice',axis=1)

x.shape

q=pd.concat([x,test])#объединенный датасет из train и test

a=q.isnull().sum()

b=a.to_frame()#датафрейм, где индекс-названия колонок, а другая колонка-кол-во пропущенных значений в этих колонках

t=list(b.loc[:,0])#кол-во пропущенных значений

f=list()#сюда буду записывать названия колонок, где есть пропущенные значения

for i in range(len(b)):

    if t[i]>0:

        f.append(b.index[i])

print(f)
e=['Alley',

 'MasVnrType',

 'BsmtQual',

 'BsmtCond',

 'BsmtExposure',

 'BsmtFinType1',

 'BsmtFinType2',

 'FireplaceQu',

 'GarageType',

 'GarageFinish',

 'GarageQual',

 'GarageCond',

 'PoolQC',

 'Fence',

 'MiscFeature']#колонки, где значения пропущенных значений оговорены в описании данных, и их использование уместно

q[e]=q[e].fillna('NA')

w=['BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath']#колонки,связанные с колонками Bsmt, где есть значения NA

q[w]=q[w].fillna(0)

q['MasVnrArea']=q['MasVnrArea'].fillna(0)#NA в этой колонке=none в колонке MasVnrType

q['LotFrontage']=q['LotFrontage'].fillna(q['LotFrontage'].median())#предпочел заполнение медианой,так же можно заполнить средним значением, модой,заполнить с помощью KNN или сгруппировать по значениям какой-нибудь колонки(например,  Neighbourhoud)

q['GarageYrBlt']=q['GarageYrBlt'].fillna(0)#пропущенные значения в этой колонке, отсутствие гаража, поэтоу я решил заполнить пропущенные значения нулем(отсутствие гаража хуже нежели старый гараж)

q[['GarageCars','GarageArea']]=q[['GarageCars','GarageArea']].fillna(0)#отсутствие гаража

r=['MSZoning',

 'Utilities',

 'Exterior1st',

 'Exterior2nd',

 'Electrical',

 'KitchenQual',

 'Functional',

 'SaleType']

q[r]=q[r].fillna(dict(q[r].mode().iloc[0]))

print(q.isna().sum().sum())#проверка, что пропущенных значений не осталось
q['MSSubClass']=q['MSSubClass'].map(str)#эта фича больше номинативная

q=pd.get_dummies(q.drop('Id',axis=1))#Делаем one-hot encoding,дабы привести данные в формат, понятный алгоритму

xtrain,xtest=q.iloc[:1460],q.iloc[1460:]#обратно разбиваем на изначальные train и test

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

y=data['SalePrice']

xtr,xval,ytr,yval=train_test_split(xtrain,y,test_size=0.3,random_state=42)#ради проверки и подбора

GBR= GradientBoostingRegressor(n_estimators=350,random_state=42)#параметры можно подобрать через Randomized(или Grid)SearchCV, но я предпочел подобрать на валидационной выборке

GBR.fit(xtr,ytr)

from sklearn.metrics import mean_squared_log_error as error

y_pr_val=GBR.predict(xval)

print(error(yval,y_pr_val))
yts=GBR.predict(xtest)

d = {'Id': range(1461,2920), 'SalePrice': list(yts)}

submission=pd.DataFrame(d)

submission.to_csv('mysubmission.csv',index=False)