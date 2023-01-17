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
import seaborn as sns
data=pd.read_csv("..//input//housesalesprediction//kc_house_data.csv")
data.describe()
data.info()

data=data.drop(['id','date'],axis=1)

data.head()
data.isnull().any()
a=sns.pairplot(data[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors']],palette='tab20')

numeric_data=data._get_numeric_data().columns
numeric_data=data[numeric_data]
corr=numeric_data.corr(method='pearson')

corr
corr_price=corr.iloc[:1,:]
col=[]

for colu,colval in corr_price.iteritems():

    if float(colval)>=0.3 or float(colval)<=-0.5:

        col.append(colu)

x_col=col[1:]

X=data[x_col]

Y=data['price']
X,Y
from sklearn.model_selection import train_test_split



x, x_test, y, y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8)
from sklearn.linear_model import LinearRegression

reg=LinearRegression(normalize='True')

reg.fit(x,y)
y_pred=reg.predict(x_test)
import matplotlib.pyplot as plt 

df=pd.DataFrame({'Actual': y_test,'Predicted':y_pred})

df1=df.head(100)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major',linestyle='-', linewidth='0.5',color='green')

plt.grid(which='major',linestyle='-',linewidth='0.5',color='black')

plt.show()
from sklearn import metrics

import numpy as np



MAE=metrics.mean_absolute_error(y_test,y_pred)

MSE=metrics.mean_squared_error(y_test,y_pred)



print("Mean Absolute Error",+MAE)

print("Mean Squared Error",+ np.sqrt(MSE))

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

polyfeat = PolynomialFeatures(degree =2)

poly_reg = linear_model.LinearRegression()

poly_reg.fit(x,y)

y_pred_poly= poly_reg.predict(x_test)
import matplotlib.pyplot as plt 

df=pd.DataFrame({'Actual': y_test,'Predicted':y_pred_poly})

df1=df.head(100)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major',linestyle='-', linewidth='0.5',color='green')

plt.grid(which='major',linestyle='-',linewidth='0.5',color='black')

plt.show()
MAE_poly=metrics.mean_absolute_error(y_test,y_pred_poly)

MSE_poly=metrics.mean_squared_error(y_test,y_pred_poly)



print("Mean Absolute Error",+MAE_poly)

print("Mean Squared Error",+ np.sqrt(MSE_poly))
