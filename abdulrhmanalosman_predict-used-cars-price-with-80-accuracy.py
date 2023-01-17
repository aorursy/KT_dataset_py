

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#reading data

data=pd.read_csv('../input/used-cars-database/autos.csv',encoding = "ISO-8859-1")



data.head()
data.shape
data=data.drop(['seller','offerType','dateCrawled','dateCreated','lastSeen','nrOfPictures','postalCode'],axis=1)

print(data.shape)

data.head()
data.describe()
#control if we have missing data (nan) in dataset 

data.isnull().sum().to_frame('nulls')
# complete missing data with most frequncy value

model=data["model"].value_counts()

print(model)

data["model"].fillna("golf",inplace=True)
# complete missing data with most frequncy value

fuelType=data["fuelType"].value_counts()

print(fuelType)

data["fuelType"].fillna("benzin",inplace=True)
# complete missing data with most frequncy value

notRepairedDamage=data["notRepairedDamage"].value_counts()

print(notRepairedDamage)

data["notRepairedDamage"].fillna("nein",inplace=True)
# We will use the Interpolation method in gearbox and vehicleType columns, (I'll explain the reason later.)

#but  to use the Interpolation method, we need to convert gearbox and vehicleType columns from object to category.



data= data.astype({"name":'category',"abtest":'category',"vehicleType":'category',"gearbox":'category',"model":'category',"fuelType":'category',"brand":'category',"notRepairedDamage":'category'}) 

data.dtypes
gearbox=data["gearbox"].value_counts()

print(gearbox)
data["gearbox"]=(data["gearbox"].cat.codes.replace(-1, np.nan).interpolate().astype(int).astype('category').cat.rename_categories(data["gearbox"].cat.categories))

vehicleType=data["vehicleType"].value_counts()

print(vehicleType)
data["vehicleType"]=data["vehicleType"].interpolate(method='pad')

#Delete line 1 min./From the interpolation process, only the nan in the first line remained. So I deleted it.

data=data.drop([0],axis=0)
price=pd.DataFrame(data.price.unique())

price.head(10)
print(len(data))

data=data[(data.price > 100) & (data.price < 200000) ]

print(len(data))
data.isnull().sum().to_frame('nulls')
#we will do LabelEncoder to feature data(x) .

x_data=data.drop('price',axis=1)

#label encoding

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

encoded_data=x_data.apply(LabelEncoder().fit_transform) #tüm veriler encod edildi 

encoded_data.head()
#Make StandardScaler to feature data(x) data

from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

X=pd.DataFrame(sc1.fit_transform(encoded_data))

X.head()
x = X.iloc[:,:].values       #features (x)

y= data.iloc[:,1:2].values   #target (price)
# splitting dataset to train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33, random_state = 0)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(x_train, y_train)
#algoritmeyi test et

y_pred=rfr.predict(x_test)

print('score: %.2f' % rfr.score(x_test, y_test))  #score 
#here we will do a comparison between real and predicted data.

print('Real Data')

print(pd.DataFrame(y_test).head(10))

print('predicted Data')

print(pd.DataFrame(y_pred).head(10))

#p value hesaplama

x=pd.DataFrame(x)

import statsmodels.api as sm

X=np.append(arr=np.ones((len(x),1)).astype(int),values=x,axis=1)

X_l=x.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]].values

r_ols=sm.OLS(endog=y,exog=X_l)

r=r_ols.fit()

print(r.summary())