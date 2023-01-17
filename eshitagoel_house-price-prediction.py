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
df = pd.read_csv("/kaggle/input/real-estate-price-prediction/Real estate.csv")
df = pd.DataFrame(df)
df.head()
df.drop(columns = "No",inplace = True)
df.head()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.heatmap(df.corr())
df.rename(columns = {"X1 transaction date" : "TransDate","X2 house age":"Age", "X3 distance to the nearest MRT station":"StatDist","X4 number of convenience stores":"No_Stores","X5 latitude":"Latitude","X6 longitude":"Longitude","Y house price of unit area":"Price"} ,inplace = True)
df.head()
df.dtypes
df.describe()
plt.figure(figsize = (15,10))
plt.subplot(2,3,1)
sns.regplot("TransDate","Price",data = df)
plt.subplot(2,3,2)
sns.regplot("Age","Price",data = df)
plt.subplot(2,3,3)
sns.regplot("StatDist","Price",data = df)
plt.subplot(2,3,4)
sns.regplot("No_Stores","Price",data = df)
plt.subplot(2,3,5)
sns.regplot("Latitude","Price",data = df)
plt.subplot(2,3,6)
sns.regplot("Longitude","Price",data = df)

df.head()
df.isnull()
X = df[["Age","StatDist","No_Stores","Latitude","Longitude"]]
Y = df[["Price"]]
X = np.asanyarray(X)
Y = np.asanyarray(Y)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scale = sc.fit_transform(X)
X_scale
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scale,Y,test_size = 0.12,random_state = 1)

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train,y_train)
print(regr.coef_)
print(regr.intercept_)
yhat = regr.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test,yhat)
# We try Polynomial Regression to check if that is any more accurate

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X_scale)
X_poly
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_poly,Y,test_size = 0.1,random_state = 1)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
yhat2 = lm.predict(X_test)
r2_score(y_test,yhat2)
