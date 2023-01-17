import numpy as np

import pandas as pd 

import os

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import plotly.offline as py

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

globalTemp=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv')

countryTemp=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')

cityTemp=pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')
countryTemp.head().T
print(countryTemp.shape[0])



print(countryTemp.columns.tolist())



print(countryTemp.dtypes)
countryTemp.Country.value_counts()
countryTemp.groupby('Country').mean()
countryTemp.groupby('Country').max()
countryTemp.groupby('Country').median()
ax = plt.axes()



ax.scatter(countryTemp.AverageTemperature, countryTemp.AverageTemperatureUncertainty)



ax.set(xlabel='AverageTemperature',

       ylabel='AverageTemperatureUncertainty',

       title='Değerler');


plt.hist(countryTemp.AverageTemperature, bins=10)

plt.xlabel("Sıcaklıklar")

plt.ylabel("Miktar")
import seaborn as sns

sns.set_context('notebook')



ax = countryTemp.plot.hist(bins=25, alpha=0.5)

ax.set_xlabel('Size (cm)');
globalTemp=globalTemp[['dt','LandAverageTemperature']]

globalTemp.dropna(inplace=True)

globalTemp['dt']=pd.to_datetime(globalTemp.dt).dt.strftime('%d/%m/%Y')

globalTemp['dt']=globalTemp['dt'].apply(lambda x:x[6:])

globalTemp=globalTemp.groupby(['dt'])['LandAverageTemperature'].mean().reset_index()

trace=go.Scatter(

    x=globalTemp['dt'],

    y=globalTemp['LandAverageTemperature'],

    mode='lines',

    )

data=[trace]



py.iplot(data, filename='line-mode')