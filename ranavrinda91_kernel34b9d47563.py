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
import pandas as pd

df  = pd.read_csv('/kaggle/input/indian-startup-funding/startup_funding.csv')

df.head(6)
percent_missing = df.notna().sum() / len(df)*100

percent_missing.sort_values()
X=df[['Industry Vertical','SubVertical','InvestmentnType',]]

#Amount is USD coantins the data in string format. we need to clean the data and convert it into int

#CHECKING NULL VALUES IN AMOUNT COLOUMN

df['Amount in USD']= df['Amount in USD'].str.replace(',', '')

df['Amount in USD']= df['Amount in USD'].str.replace('+', '')

df.loc[df['Amount in USD'] == 'undisclosed', 'Amount in USD'] = 0

df.loc[df['Amount in USD'] == 'Undisclosed', 'Amount in USD'] = 0

df['Amount in USD']=df['Amount in USD'].str.replace(r'[\\xc2\\xa020000000]', '0')

df['Amount in USD']=df['Amount in USD'].str.replace(r'[0000000000N/A]', '0')

df['Amount in USD']=pd.to_numeric(df['Amount in USD'])

#filling all the NAN values with 0

df['Amount in USD']=df['Amount in USD'].fillna(0)

y=df['Amount in USD']

y.tail(3)
#converting the text feature into numeric values

XX =X.apply(lambda col: pd.factorize(col, sort=True)[0])

# Filling all the categorical null values with the mode

# we can fill them with the medain or mode too 

XX['InvestmentnType'] = XX['InvestmentnType'].fillna((XX['InvestmentnType'].mode()))

XX['Industry Vertical'] = XX['Industry Vertical'].fillna((XX['Industry Vertical'].mode()))

XX['SubVertical'] = XX['SubVertical'].fillna((XX['SubVertical'].mode()))

XX.head(3)

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression



# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.3) # 70% training and 30% test



#Create a Gaussian Classifier

regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=200)

#Tra in the model using the training sets y_pred=clf.predict(X_test)

model =regr.fit(X_train,y_train)

print(regr.feature_importances_)

y_pred=regr.predict(X_test)



print("Number of predictions:",len(y_pred))

meanSquaredError=mean_squared_error(y_test, y_pred)

print("MSE:", meanSquaredError)

rootMeanSquaredError = sqrt(meanSquaredError)

print("RMSE:", rootMeanSquaredError)