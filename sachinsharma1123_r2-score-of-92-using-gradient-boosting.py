# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv')
df
df.isnull().sum()
df.corr()['price']
df['model'].unique()
df['fuelType'].unique()
import matplotlib.pyplot as plt

plt.scatter(df['mileage'],df['price'])

plt.show()
#this plot shows that the price of a car decreases as the mileage increases
plt.scatter(df['engineSize'],df['price'])

plt.show()
from sklearn.preprocessing import LabelEncoder

Model=pd.get_dummies(df['model'],drop_first=True)
Model
df=df.drop(['model'],axis=1)
df=pd.concat([df,Model],axis=1)
df
df=df.drop(['year'],axis=1)
le=LabelEncoder()

df['transmission']=le.fit_transform(df['transmission'])
df['fuelType']=le.fit_transform(df['fuelType'])
df
x=df.drop(['price'],axis=1)
y=df['price']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

pred_y=lr.predict(x_test)
from sklearn.metrics import r2_score

print(r2_score(y_test,pred_y))
from sklearn.linear_model import Lasso

la=Lasso()

la.fit(x_train,y_train)

pred_2=la.predict(x_test)

print(r2_score(y_test,pred_2))
from sklearn.linear_model import Ridge

rd=Ridge()

rd.fit(x_train,y_train)

pred_3=rd.predict(x_test)

print(r2_score(y_test,pred_3))
from sklearn.model_selection import GridSearchCV

params_lasso={'alpha':[1,0.1,0.01,0.001,0.0001]}

params_ridge={'alpha':[0.1,0.01,1,2,5,10,20,50,75,100,150,200,250,300,400,450,500,550,600]}

reg=GridSearchCV(rd,params_ridge,verbose=0)

model=reg.fit(x_train,y_train)
print(model.best_params_)
rd=Ridge(alpha=0.1)

rd.fit(x_train,y_train)

pred_4=rd.predict(x_test)

print(r2_score(y_test,pred_4))
reg_1=GridSearchCV(la,params_lasso,verbose=0)

reg_1.fit(x_train,y_train)

print(reg_1.best_params_)
la=Lasso(alpha=0.1)

la.fit(x_train,y_train)

pred_5=la.predict(x_test)

print(r2_score(y_test,pred_5))
from sklearn.ensemble import GradientBoostingRegressor
rgb=GradientBoostingRegressor()

rgb.fit(x_train,y_train)

pred_6=rgb.predict(x_test)

print(r2_score(y_test,pred_6))
new_df=pd.DataFrame({'actual':y_test.values.flatten(),

                    'predicted':pred_6})

new_df
plt.scatter(y_test.values.flatten(),pred_6)

plt.show()