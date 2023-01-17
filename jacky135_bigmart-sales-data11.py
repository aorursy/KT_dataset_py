# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/bigmart-sales-data/Train.csv')

test = pd.read_csv('/kaggle/input/bigmart-sales-data/Test.csv')
train.info()
y= train['Item_Outlet_Sales']
train['Item_Identifier'] = train['Item_Identifier'].apply(lambda x:x[0:2])

test['Item_Identifier'] = test['Item_Identifier'].apply(lambda x:x[0:2])



train['Item_Weight'].fillna(train['Item_Weight'].mean(),inplace = True)

test['Item_Weight'].fillna(test['Item_Weight'].mean(),inplace = True)
train['Outlet_Size'].fillna("Medium",inplace = True)

test['Outlet_Size'].fillna("Medium",inplace = True)
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})

test['Item_Fat_Content'] = test['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})
train['Outlet_Establishment_Year'].replace({1999:"1999",1998:"1998",2009:"2009",1985:"1985",1987:"1987",2002:"2002",2007:"2007",1997:"1997",2004:"2004"})

test['Outlet_Establishment_Year'].replace({1999:"1999",1998:"1998",2009:"2009",1985:"1985",1987:"1987",2002:"2002",2007:"2007",1997:"1997",2004:"2004"})
dum = pd.get_dummies(data = train,columns = ['Outlet_Establishment_Year','Item_Fat_Content','Item_Identifier','Outlet_Location_Type','Outlet_Size','Item_Type','Outlet_Type','Outlet_Identifier'])

dum1 = pd.get_dummies(data = test,columns = ['Outlet_Establishment_Year','Item_Fat_Content','Item_Identifier','Outlet_Location_Type','Outlet_Size','Item_Type','Outlet_Type','Outlet_Identifier'])
train.info()
dum.info()
dum.drop('Item_Outlet_Sales',axis = 1,inplace = True)
dum.shape
dum1.shape
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(dum)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state=42)
from sklearn import neighbors

from sklearn.metrics import mean_squared_error 

from math import sqrt

rmse_val = []

for K in range(30):

    model = neighbors.KNeighborsRegressor(n_neighbors = K+1)

    model.fit(X_train, y_train) 

    pred = model.predict(X_test)

    error = sqrt(mean_squared_error(y_test,pred))

    print(error)

    rmse_val.append(error)

    
curve = pd.DataFrame(rmse_val)

curve
curve.plot()