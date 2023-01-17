import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', index_col='Country/Region')
# group & sum by country, coz bigger countries are split to regions
cases = confirmed_df.groupby('Country/Region').sum()
cases.head()
# cases.loc['Czechia']
populations_data = pd.read_csv('../input/countries-population-2020/populations.csv')
populations = populations_data.groupby('Location').sum()
cases.insert(loc=0, column='Population', value = populations['PopTotal'])
cases = cases.loc[cases['Population'] > 1000]
N = 7
cases['NdayIncrement'] = (cases.iloc[:,-1] - cases.iloc[:,-1-N])
cases['NdayAverage'] = cases['NdayIncrement'] / N
cases['NdayIncrementPerPopulation'] = cases['NdayIncrement'] / cases['Population'] * 100
cases['NdayAveragePerPopulation'] = cases['NdayAverage'] / cases['Population'] * 100
result = cases.loc[:,['Population', 'NdayIncrement', 'NdayAverage', 'NdayIncrementPerPopulation' ,'NdayAveragePerPopulation']]
top = result.sort_values(by=['NdayAveragePerPopulation'], ascending=False).head(50)
top
top.rename(columns = {'NdayAveragePerPopulation':'Cases Count'}, inplace = True)
top.plot(kind='bar', y='Cases Count', figsize=(10,8))