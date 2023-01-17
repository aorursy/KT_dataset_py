import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

data1= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data1
data2= pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
data2
data1.isnull().sum()
data1.isnull().sum()

pd.set_option('display.max_rows',81)

pd.set_option('display.max_columns',81)
data1
data1.isnull().sum()
data1
data1.corr()
data1['LandContour'].unique()
a=data1.drop(['Id','Alley','HouseStyle','FireplaceQu','Exterior2nd','Condition2','ExterQual','GrLivArea','BsmtFinType2','BsmtFinSF2','BsmtHalfBath','GarageYrBlt','GarageCars','GarageCond','3SsnPorch','PoolQC','Fence','MiscVal','MiscFeature','MoSold','YrSold'],axis=1)
a
a.isnull().sum()
b=a.dropna(subset=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','Electrical'])

b
b.isnull().sum()
b
b.isnull().sum()
c=b.fillna({'GarageFinish':'RFn','GarageQual':'TA','GarageType':'Attchd','MasVnrType':'None'})
c
c.isnull().sum()
x=c.iloc[:,:-1].values

y=c.iloc[:,-1].values
from sklearn.impute import SimpleImputer

imp=SimpleImputer(copy=True, missing_values=np.nan, strategy='mean', verbose=0)

x[:,[2]]=imp.fit_transform(x[:,[2]])

x[:,[21]]=imp.fit_transform(x[:,[21]])
x
d=pd.DataFrame(x)

d
from sklearn.preprocessing import LabelEncoder

label_x=LabelEncoder()

x[:,1]=label_x.fit_transform(x[:,1])

x[:,4]=label_x.fit_transform(x[:,4])

x[:,5]=label_x.fit_transform(x[:,5])

x[:,6]=label_x.fit_transform(x[:,6])

x[:,7]=label_x.fit_transform(x[:,7])

x[:,8]=label_x.fit_transform(x[:,8])

x[:,9]=label_x.fit_transform(x[:,9])

x[:,10]=label_x.fit_transform(x[:,10])

x[:,11]=label_x.fit_transform(x[:,11])

x[:,12]=label_x.fit_transform(x[:,12])

x[:,17]=label_x.fit_transform(x[:,17])

x[:,18]=label_x.fit_transform(x[:,18])

x[:,19]=label_x.fit_transform(x[:,10])

x[:,20]=label_x.fit_transform(x[:,20])

x[:,22]=label_x.fit_transform(x[:,22])

x[:,23]=label_x.fit_transform(x[:,23])

x[:,24]=label_x.fit_transform(x[:,24])

x[:,25]=label_x.fit_transform(x[:,25])

x[:,26]=label_x.fit_transform(x[:,26])

x[:,27]=label_x.fit_transform(x[:,27])

x[:,31]=label_x.fit_transform(x[:,31])

x[:,32]=label_x.fit_transform(x[:,32])

x[:,33]=label_x.fit_transform(x[:,33])

x[:,34]=label_x.fit_transform(x[:,34])

x[:,43]=label_x.fit_transform(x[:,43])

x[:,45]=label_x.fit_transform(x[:,45])

x[:,47]=label_x.fit_transform(x[:,47])

x[:,48]=label_x.fit_transform(x[:,48])

x[:,50]=label_x.fit_transform(x[:,50])

x[:,51]=label_x.fit_transform(x[:,51])

x[:,57]=label_x.fit_transform(x[:,57])

x[:,58]=label_x.fit_transform(x[:,58])
x
x[:,58]
x[:,45]
x[:,18]
x[:,10]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(

     x, y, test_size=0.2, random_state=42)
x_train
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

y_pred
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
data2
data2.isnull().sum()
d=data2.drop(['Id','Alley','HouseStyle','FireplaceQu','Exterior2nd','Condition2','ExterQual','GrLivArea','BsmtFinType2','BsmtFinSF2','BsmtHalfBath','GarageYrBlt','GarageCars','GarageCond','3SsnPorch','PoolQC','Fence','MiscVal','MiscFeature','MoSold','YrSold'],axis=1)
d
d.isnull().sum()
e=d.fillna({'GarageFinish':'RFn','GarageQual':'TA','GarageType':'Attchd','MasVnrType':'None','MSZoning':'RH','Utilities':'AllPub','Exterior1st':'VinylSd','BsmtQual':'TA','BsmtCond':'TA','BsmtFinType1':'GLQ','BsmtExposure':'No','KitchenQual':'TA','Functional':'Typ','SaleType':'WD'})
e.isnull().sum()
e
l=e.iloc[:].values
l
from sklearn.impute import SimpleImputer

imp1=SimpleImputer(missing_values=np.nan, strategy='mean')

l[:,[2]]=imp.fit_transform(l[:,[2]])

l[:,[21]]=imp.fit_transform(l[:,[21]])

l[:,[28]]=imp.fit_transform(l[:,[28]])

l[:,[29]]=imp.fit_transform(l[:,[29]])

l[:,[30]]=imp.fit_transform(l[:,[30]])

l[:,[49]]=imp.fit_transform(l[:,[49]])
l
imp2=SimpleImputer(missing_values=np.nan,strategy="most_frequent")

l[:,[38]]=imp2.fit_transform(l[:,[38]])
g=pd.DataFrame(l)
g
g.isnull().sum()
from sklearn.preprocessing import LabelEncoder

label_x=LabelEncoder()

l[:,1]=label_x.fit_transform(l[:,1])

l[:,4]=label_x.fit_transform(l[:,4])

l[:,5]=label_x.fit_transform(l[:,5])

l[:,6]=label_x.fit_transform(l[:,6])

l[:,7]=label_x.fit_transform(l[:,7])

l[:,8]=label_x.fit_transform(l[:,8])

l[:,9]=label_x.fit_transform(l[:,9])

l[:,10]=label_x.fit_transform(l[:,10])

l[:,11]=label_x.fit_transform(l[:,11])

l[:,12]=label_x.fit_transform(l[:,12])

l[:,17]=label_x.fit_transform(l[:,17])

l[:,18]=label_x.fit_transform(l[:,18])

l[:,19]=label_x.fit_transform(l[:,10])

l[:,20]=label_x.fit_transform(l[:,20])

l[:,22]=label_x.fit_transform(l[:,22])

l[:,23]=label_x.fit_transform(l[:,23])

l[:,24]=label_x.fit_transform(l[:,24])

l[:,25]=label_x.fit_transform(l[:,25])

l[:,26]=label_x.fit_transform(l[:,26])

l[:,27]=label_x.fit_transform(l[:,27])

l[:,31]=label_x.fit_transform(l[:,31])

l[:,32]=label_x.fit_transform(l[:,32])

l[:,33]=label_x.fit_transform(l[:,33])

l[:,34]=label_x.fit_transform(l[:,34])

l[:,43]=label_x.fit_transform(l[:,43])

l[:,45]=label_x.fit_transform(l[:,45])

l[:,47]=label_x.fit_transform(l[:,47])

l[:,48]=label_x.fit_transform(l[:,48])

l[:,50]=label_x.fit_transform(l[:,50])

l[:,51]=label_x.fit_transform(l[:,51])

l[:,57]=label_x.fit_transform(l[:,57])

l[:,58]=label_x.fit_transform(l[:,58])
l
l[:,10]
id=data2['Id']
id
from sklearn.ensemble import GradientBoostingRegressor

gbc=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1)
gbc
model2=gbc.fit(x_train,y_train)
ypred2=model2.predict(x_test)
r2_score(y_test,ypred2)
ypred3=model2.predict(l)
ypred3
SalePrice=pd.DataFrame(ypred3)
SalePrice
x = id

y = SalePrice

style.use('dark_background')



plt.plot(x,y)



plt.title('id vs Sales Price')

plt.ylabel('Y axis')

plt.xlabel('X axis')

plt.show()
submission=pd.concat([id,SalePrice],axis=1)

submission.columns=['Id','SalePrice']

submission.to_csv('submission.csv',index=False)
submission