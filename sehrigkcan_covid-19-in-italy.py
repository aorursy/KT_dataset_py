# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/covid19-in-italy/covid19_italy_region.csv")
data.tail(20)
data.describe().T
data.corr()
data.info()

sns.pairplot(data,kind="reg");

data
# Polinom

# Suanki Pozitif Olan Kisi Sayisi ile Kurtarilanlar Arasindaki Polinom Iliski 

df = data.groupby('Date')[['Country', 'HospitalizedPatients', 'IntensiveCarePatients',
       'TotalHospitalizedPatients', 'HomeConfinement', 'CurrentPositiveCases',
       'NewPositiveCases', 'Recovered', 'Deaths', 'TotalPositiveCases',
       'TestsPerformed']].sum()

CurrentPositiveCases= df.iloc[:,4:5].values.reshape(-1,1)
recorved= df.iloc[:,6:7].values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
lr = LinearRegression()

poly = PolynomialFeatures(degree=6)

CurrentPositiveCases_poly = poly.fit_transform(CurrentPositiveCases)

lr.fit(CurrentPositiveCases_poly, recorved)

predict = lr.predict(CurrentPositiveCases_poly)

plt.scatter(CurrentPositiveCases, recorved, color='red')
plt.plot(CurrentPositiveCases, predict, color='blue')
plt.show()
# Linear Iliski 

#Toplam Testi Pozitif Cikan Kisi Sayisi  ile Vefat Edenler Arasindaki Linear Iliski 

TotalPositiveCases= df.iloc[:,8:9].values.reshape(-1,1)
Deaths= df.iloc[:,7:8].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(TotalPositiveCases,Deaths,test_size=0.33,random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)
x_train=np.sort(x_train)
y_train=np.sort(y_train)

plt.scatter(x_train,y_train)
plt.plot(x_test,tahmin,color="red")
plt.show()
df.columns
#Multiple 


# Yogun Bakim Hastalari-Hastanedeki Hastalar ve Evde Kalan Hastalar ile Kurtarilabilen Hastalar Arasindaki Iliski 

#IntensiveCarePatients
#HospitalizedPatients
#HomeConfinement 

pre=pd.concat([df.iloc[:,0:1],df.iloc[:,1:2],df.iloc[:,3:4]],axis=1)
lr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(pre,recorved,test_size=0.33,random_state=0)
lr.fit(x_train,y_train)
result = lr.predict(x_test)
#print(result)

plt.scatter(result,y_test)
plt.plot(result,y_test)
plt.show()
import statsmodels.regression.linear_model as sm

deaths= df.iloc[:,7:8].values

X = np.append(arr=np.ones((len(df.Deaths),1)).astype(int),values=df,axis=1)

X_l = df.iloc[:,:]

r = sm.OLS(endog=deaths,exog=X_l).fit()

r.summary()