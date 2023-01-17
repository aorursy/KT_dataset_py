# import module 

import numpy as np 

import seaborn as sns 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 
df=pd.read_csv('../input/kc_house_data.csv')
df.shape
df.isnull().any() # check any null data 
df.describe() # check data scale 
df.head() # check data range 
# common sense, the id and date should not be related with price. 

# the rest property should be related with price, there are two column data might need some transfrom

# first : check the data distribution of yr_renovated whether too sparse

# second is zip, lat, lon, could be integrated into location. 



# check value of yr_renovated 

df.yr_renovated.value_counts()

price=df.price;yr_renovated=df.yr_renovated

plt.scatter(yr_renovated,price)
# lets check the heatmap between each column 

plt.figure(figsize=(8,6))

sns.heatmap(df.corr(),cmap='bone',annot=True,fmt='0.1f',)
# so the correlation among price and location(zipcode,lat,lon),sqft_lot15,and condition et al is 

# low , while this might due to the relative scale of this column is too small. 

# so will try use lasso to learning the data, the decide if we want to remove some property 

y=np.array(df['price'])

df=df.drop('price',axis=1)

df=df.drop('id',axis=1)

df=df.drop('lat',axis=1)

df=df.drop('long',axis=1)

df=df.drop('date',axis=1)
df.head()
x=np.array(df)
# import lasso 

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn import linear_model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
sc=StandardScaler()

sc.fit(x_train)

x_train_std=sc.transform(x_train)

sc.fit(x_test)

x_test_std=sc.transform(x_test)
LM=linear_model.Lasso(alpha=0.1,max_iter=3000)
LM.fit(x_train_std,y_train)
prediction=LM.predict(x_test_std)

predit=prediction.reshape(-1,1)

y_t=y_test.reshape(-1,1)
LM.score(x_test_std,y_t)
# 

LSM=linear_model.Ridge(alpha=0.8)
LSM.fit(x_train_std,y_train)
LSM.score(x_test_std,y_test)
# it's too low, lets check if there is any outlier 

import seaborn as sns
df['price']=y
df.head()
x=np.arange(0,21613,1)
df[df.bedrooms>9]
fig=plt.figure(figsize=(100,100))

#fig.set_figheight(20)

#fig.set_figwidth(20)
plt.subplot(1,1,1)

plt.scatter(x,df.iloc[:,1])

#plt.subplot(5,2,2)

#plt.scatter(x,df.iloc[:,2])

#plt.subplot(5,2,3)

#plt.scatter(x,df.iloc[:,3])
df[df.bedrooms>9]
# records that bedroom is 33 with 1.75 bathroom should be wrong. so change bedroom number into 3 

df.ix[15870,1]=3
title=df.columns

for i in range(df.shape[1]):

    fig=plt.figure(num=i,figsize=(3,4))

    plt.scatter(x,df.iloc[:,i])

    plt.title(title[i])
df[df.bathrooms>6]

# looks ok
df[df.sqft_living>10000]

# the lot size for id 1225069038 looks not reasonalbe, its build on 1999, and have large house, but way 

# more cheaper, in case, i will drop this row 

df2=df.drop(12777)
df2[df2.sqft_above>8000] # looks fine 

df2[df2.sqft_basement>3000] # also looks fine 

# after correct data

#redo the regressions

y=np.array(df2['price'])

x=np.array(df2)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

sc.fit(x_train)

x_train_std=sc.transform(x_train)

sc.fit(x_test)

x_test_std=sc.transform(x_test)
y_train=y_train[:,np.newaxis]

y_test=y_test[:,np.newaxis]
sc.fit(y_train)

y_train_std=sc.transform(y_train)
lm=linear_model.Lasso(alpha=0.5)
lm.fit(x_train_std,y_train_std)
sc.fit(y_test)

y_test_std=sc.transform(y_test)
lm.score(x_test_std,y_test_std)
# the score is imporved by 12 percentage

# so next step try some other methods, like nerual network

from sklearn.neural_network import MLPRegressor

from sklearn.cross_validation import column_or_1d

x_test_std.shape
mlp=MLPRegressor(hidden_layer_sizes=(30,17),activation='relu',alpha=0.01)
mlp.fit(x_train_std,column_or_1d(y_train_std))
prediction=mlp.predict(x_test_std)
mlp.score(x_test_std,y_test_std)