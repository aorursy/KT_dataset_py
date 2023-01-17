import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

from scipy import stats

import plotly.express as px

from sklearn.preprocessing import LabelEncoder

from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot

from fbprophet.diagnostics import cross_validation, performance_metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
def overview():

    '''

    Read a comma-separated values (csv) file into DataFrame.

    Print 5 rows of data

    Print number of rows and columns

    Print datatype for each column

    Print number of NULL/NaN values for each column

    Print summary data

    

    Return:

    data, rtype: DataFrame

    '''

    data = pd.read_csv('../input/avocado-prices/avocado.csv')

    print("The first 5 rows if data are:\n", data.head())

    print("\n")

    print("The (Row,Column) is:\n", data.shape)

    print("\n")

    print("Data type of each column:\n", data.dtypes)

    print("\n")

    print("The number of null values in each column are:\n", data.isnull().sum())

    print("\n")

    print("Summary of all the test scores:\n", data.describe())

    return data



df = overview()
df = df.rename(columns={'4046': "Small_Hass", '4225': "Large_Hass", '4770': "Extra_Large_Hass"})

df = df.drop(columns=['Unnamed: 0'])
plt.figure(figsize=(10,5))



sns.distplot(df['AveragePrice'])
sns.boxplot(data = df, x = 'AveragePrice')
plt.figure(figsize=(15,10))

sns.barplot(x = 'year', y = 'AveragePrice', hue = 'type', data = df)
plt.figure(figsize=(15,10))



g = sns.factorplot('AveragePrice','region', data = df, hue='year', height = 13, aspect = 0.8, palette ='husl', join = False)
le = LabelEncoder()



# Implementing LE on type

le.fit(df.type.drop_duplicates()) 

df.type = le.transform(df.type)



# Implementing LE on regions

le.fit(df.region.drop_duplicates()) 

df.region = le.transform(df.region)
plt.figure(figsize=(15,10))





corrMatrix = df.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
# Here, we will only need columns that contain the timestamp and the measure that we will like to predict 

cols = ['Date', 'AveragePrice']

df1 = df[cols]

df1.columns = ['ds', 'y']



m = Prophet()

m.fit(df1)



future = m.make_future_dataframe(periods=365)



forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)
# Getting rid outliers

q = df['AveragePrice'].quantile(0.99)

df = df[df['AveragePrice']< q]





y = df['AveragePrice']

X = df[['Small_Hass', 'Large_Hass', 'Extra_Large_Hass', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year', 'region']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





lr = LinearRegression().fit(X_train,y_train)

y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



print(lr.score(X_test,y_test))