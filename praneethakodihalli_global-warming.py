# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/who-is-resposible-for-global-warming/API_EN.ATM.CO2E.PC_DS2_en_csv_v2_10576797.csv')
df.describe()
df.head()
li=[]
i=1960
for i in range(1960,2015):
    li.append(df[str(i)].mean())
print(li)
lii=[str(i) for i in range(1960,2015)]
print(lii)
data={'year':lii,'avg_co2emissions':li}
d = pd.DataFrame(data)
d
maximumavg=max(d['avg_co2emissions'])
print(maximumavg)
x=d.index[d['avg_co2emissions']==maximumavg]
print(d.iloc[x])
d.plot(x='year', y='avg_co2emissions')
import pandas as pd
City = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv")
Country = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")
MajorCity = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByMajorCity.csv")
State = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv")
GlobalTemperatures = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv")
GlobalTemperatures.describe()
GlobalTemperatures.head()
GlobalTemperatures=GlobalTemperatures.dropna(axis=0,how='any')
GlobalTemperatures.head()
GlobalTemperatures.plot('LandMaxTemperature','LandAverageTemperature')
GlobalTemperatures.plot('LandMaxTemperature','LandMinTemperature')
India = City[City.Country == 'India']
India.head()
Hyderabad=India[City.City=='Hyderabad']
Hyderabad.head(10)
Hyderabad=Hyderabad.dropna(axis=0,how='any')
Hyderabad.dtypes
Hyderabad.shape
Hyderabad.plot(x='dt',y='AverageTemperature')
Hyderabad['year']=pd.DatetimeIndex(Hyderabad['dt']).year
Hyderabad.head()
Hyderabad.groupby('year')['AverageTemperature'].mean().plot(x='year',y='AverageTemperature')
X=Hyderabad['year'].values.reshape(-1,1)
y=Hyderabad['AverageTemperature'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)
h=pd.DataFrame([2014,2015,2016,2017,2018,2019,2020,2021,2022]).values.reshape(-1,1)
h_pred=regressor.predict(h)
h_pred
pred = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
pred
pred1 = pred.head(25)
pred1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
cities=['Hyderabad','New Delhi','Bombay','Pune','Kochi','Srinagar']
city_temp = India[India.City.isin(cities)]
city_temp.groupby('City')['AverageTemperature'].mean().plot(kind='bar')