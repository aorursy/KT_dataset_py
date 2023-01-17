# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
train.SalePrice.describe()
target=np.log(train.SalePrice)
print('skew is:',target.skew())
plt.hist(target,color='blue')
plt.show()
numeric_feat=train.select_dtypes(include=[np.number])
numeric_feat.info()
corr=numeric_feat.corr(method='pearson')
print(corr['SalePrice'].sort_values(ascending=False)[:5],'\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])
train.OverallQual.unique()
qual_table=train.pivot_table(index='OverallQual',values='SalePrice',aggfunc=np.median)
qual_table
qual_table.plot(kind='bar',color='green')
plt.xlabel('SalePrice')
plt.ylabel('overallqual')
plt.xticks(rotation=0)
plt.show()
plt.scatter(x=train.GrLivArea,y=target,color='red')
plt.xlabel('GrLivArea')
plt.ylabel('Salesprice')
plt.show()
plt.scatter(x=train.GarageArea,y=target,color='red')
plt.xlabel('garage area')
plt.ylabel('Salesprice')
plt.show()
train=train[train.GarageArea<1200]
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
nulls=pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns=['Null count']
nulls.index.name='feature'
nulls
median_val=train.LotFrontage.median()
train.LotFrontage=train.LotFrontage.fillna(median_val)
test.LotFrontage=test.LotFrontage.fillna(test.LotFrontage.median())
plt.scatter(x=train.LotFrontage,y=np.log(train.SalePrice))
#train.LotFrontage
train=train[train['GarageYrBlt'].notnull()]
train=train[train.MasVnrArea.notnull()]
test.GarageYrBlt=test.GarageYrBlt.fillna(test.GarageYrBlt.median())
test['MasVnrArea']=test.MasVnrArea.fillna(test.MasVnrArea.median())
test
categor=train.select_dtypes(exclude=[np.number])
categor.describe()
alley_piv=train.pivot_table(index='Street',values='SalePrice',aggfunc=np.median)
alley_piv.plot(kind='bar',color='blue')
train['Street']=pd.get_dummies(train.Street,drop_first=True)
test['Street'] = pd.get_dummies(test.Street, drop_first=True)

test['Street'].isnull().sum()
train
alley_piv=train.pivot_table(index='Alley',values='SalePrice',aggfunc=np.median)
alley_piv.plot(kind='bar',color='blue')
def encode(x): return 1 if x == 'Pave' else 0
train['Alley']=train.Alley.apply(encode)
test['Alley']=test.Alley.apply(encode)

train

def encode1(x): return 1 if x=='AllPub' else  0
train['Utilities']=train.Utilities.apply(encode1)
test['Utilities']=test.Utilities.apply(encode1)
def encode2(x): return 1 if x=='Y' else 0
train['CentralAir']=train.CentralAir.apply(encode2)
test['CentralAir']=test.CentralAir.apply(encode2)
train.CentralAir.value_counts()
test.CentralAir.isnull().sum()
test=pd.get_dummies(test)
train=pd.get_dummies(train)
train,test=train.align(test,join='left',axis=1)
test.isnull().sum().sort_values(ascending=False)
test.BsmtHalfBath=test.BsmtHalfBath.fillna(0)
test.BsmtFullBath=test.BsmtFullBath.fillna(0)
test.BsmtFinSF1=test.BsmtFinSF1.fillna(0)
test.TotalBsmtSF=test.TotalBsmtSF.fillna(0)
test.GarageArea=test.GarageArea.fillna(0)
test.GarageCars=test.GarageCars.fillna(0)
test.BsmtUnfSF=test.BsmtUnfSF.fillna(0)
test.BsmtFinSF2=test.BsmtFinSF2.fillna(0)
print(test.isnull().sum().sort_values(ascending=False))

def f(x):
    test[x]=test[x].replace(np.nan,0)
    

f('GarageQual_Ex')
f('MiscFeature_TenC')
f('Heating_OthW')
f('Heating_Floor')
f('Electrical_Mix')
f('Functional_Sev')
f('Condition2_RRAe')
f('Condition2_RRAn')
f('Condition2_RRNn')
f('HouseStyle_2.5Fin')
f('PoolQC_Fa')
f('Exterior2nd_Other')
f('Heating_Grav')
f('Exterior1st_ImStucc')
f('RoofMatl_Roll')
f('RoofMatl_Membran')
f('RoofMatl_Metal')
f('Exterior1st_Stone')
f('Exterior2nd_AsphShn')
f('Exterior2nd_CBlock')

test.isnull().sum().sort_values(ascending=False)
test.SalePrice
from sklearn.model_selection import train_test_split
y=np.log(train.SalePrice)
X=train.drop(['SalePrice', 'Id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
model=ensemble.GradientBoostingRegressor()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
from sklearn.metrics import mean_squared_error
pred=model.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, pred))

submission = pd.DataFrame()
submission['Id'] = test.Id
feats = test.drop(columns=['Id','SalePrice'], axis=1)
predictions = model.predict(feats)
final_predictions = np.exp(predictions)
submission['SalePrice'] = final_predictions
print(submission.shape)
submission.to_csv('submission1.csv', index=False)