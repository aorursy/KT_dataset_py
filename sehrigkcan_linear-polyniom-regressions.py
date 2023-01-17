# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
print(data.tail(20))


data.info()

data.describe().T
sns.pairplot(data,kind="reg");
plt.figure(figsize=(10,10))
sns.barplot(x=data['Last_Update'], y=data['Confirmed'])
plt.xticks(rotation= 90)
plt.xlabel('Days')
plt.ylabel('Confirmed')
# Polinom

deaths= data.iloc[:,3:4].values.reshape(-1,1)
confirmed=data.iloc[:,2:3].values.reshape(-1,1)
recorved=data.iloc[:,4:5].values.reshape(-1,1)
data["index"]=range(len(confirmed))
index=data["index"].values.reshape(-1,1)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
lr = LinearRegression()

poly = PolynomialFeatures(degree=5)

index_poly = poly.fit_transform(index)

lr.fit(index_poly, confirmed)

predict = lr.predict(index_poly)

plt.scatter(index, confirmed, color='red')
plt.plot(index, predict, color='blue')
plt.show()
#Linear 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(confirmed,recorved,test_size=0.33,random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)

x_train=np.sort(x_train)
y_train=np.sort(y_train)

plt.scatter(x_train,y_train)
plt.plot(x_test,tahmin,color="red")
plt.show()
# Multiple 

pre=pd.concat([data.iloc[:,2:3],data.iloc[:,4:5]],axis=1)
lr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(pre,deaths,test_size=0.33,random_state=0)
lr.fit(x_train,y_train)
result = lr.predict(x_test)
print(result)
import statsmodels.regression.linear_model as sm

#X = np.append(arr=np.ones((len(data),1)).astype(int),values=data,axis=1)

#X_l = data.iloc[:,[2,3,4,5]].values

#r = sm.OLS(endog=deaths,exog=X_l).fit()

# OLS sonucu konsola yazdırdık.
#print(r.summary())
