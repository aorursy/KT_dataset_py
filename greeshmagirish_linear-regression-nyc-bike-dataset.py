%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from pandas import set_option

plt.rcParams['figure.figsize'] = (25,18)
nyc_data = pd.read_csv('../input/nyc-east-river-bicycle-counts.csv',index_col='Date', parse_dates = True)

nyc_data = nyc_data.head(30)
nyc_data.head()
nyc_data =  nyc_data.drop('Day', axis = 1)

nyc_data =  nyc_data.drop('Unnamed: 0', axis = 1)

nyc_data = nyc_data.rename(columns={"High Temp (°F)": "HighTemp", "Low Temp (°F)": "LowTemp", "Precipitation	":"Precipitation"})

nyc_data.head()
nyc_data.dtypes
prep = nyc_data['Precipitation'].replace(['0.47 (S)'], '0.47')

prep = prep.replace(['T'], '0')

prep = prep.astype(float)

nyc_data['Precipitation'] = prep

nyc_data.head(4)
nyc_data.dtypes
nyc_data.describe()
nyc_data.corr()
sns.heatmap(nyc_data.corr(),annot = True)

plt.show()
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.set(style="darkgrid")

sns.relplot( x="Total", y="Precipitation", data=nyc_data);
sns.pairplot(nyc_data,x_vars=['HighTemp', 'LowTemp','Precipitation'],y_vars='Total',kind='reg',size=6)

plt.show()
nyc_data.resample('D').sum().plot()
x =nyc_data.drop(['Total','Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge'],axis=1)

y = nyc_data['Total']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=1)
linear_model = LinearRegression()

output = linear_model.fit(xtrain,ytrain)

print(output.intercept_)

output.coef_
y_pred = linear_model.predict(xtest)

np.sqrt(metrics.mean_squared_error(ytest,y_pred))
df = pd.DataFrame({})

df = pd.concat([xtest,ytest],axis=1)

df['Predicted Total'] = np.round(y_pred,2)

df['Error'] = df['Total'] - df['Predicted Total']

df