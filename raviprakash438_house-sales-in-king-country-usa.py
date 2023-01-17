# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
df.corr()['price'].sort_values(ascending=False)
#Lets plot graph for all categorical columns agains price to understand how significant they are.
cols=['sqft_living','grade','sqft_above','sqft_living15','bathrooms','view','sqft_basement','bedrooms','waterfront','floors','yr_renovated','sqft_lot',
     'sqft_lot15','yr_built','condition']

fig,axi=plt.subplots(8,2,figsize=(20,40))
indx=0
leng=len(cols)
for ax in axi:
    for a in ax:
        if leng>indx:
            a.scatter(x= cols[indx],y='price',data=df)
            a.set_title('{} vs Price'.format(cols[indx]))
            indx +=1
plt.show()

#Removing outlier from dataset.
df.loc[(df['sqft_living']>12000) & (df['price']<3000000)]
df=df.drop(df.loc[(df['sqft_living']>12000) & (df['price']<3000000)].index)
df.loc[(df['bedrooms']>30) & (df['price']<2000000)]
df=df.drop(df.loc[(df['bedrooms']>30) & (df['price']<2000000)].index)
df.loc[(df['sqft_lot']>1500000) & (df['price']<2000000)]
df=df.drop(df.loc[(df['sqft_lot']>1500000) & (df['price']<2000000)].index)
df.loc[(df['sqft_lot15']>800000) & (df['price']<2000000)]
df=df.drop(df.loc[(df['sqft_lot15']>800000) & (df['price']<2000000)].index)
y=df.price
X=df.drop(['price','date','id'],axis=1)
#X=df[['sqft_living','grade','sqft_above','sqft_living15','bathrooms','view','sqft_basement','bedrooms','waterfront','floors','yr_renovated','sqft_lot',
#     'sqft_lot15','yr_built','condition']]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=1)

reg=XGBRegressor(n_estimators=500,learning_rate=.2)
reg.fit(X_train,y_train)
pred=reg.predict(X_test)
print(mean_absolute_error(y_test,pred))
