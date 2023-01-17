import pandas as pd 
rates = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')

ex_rates = pd.read_csv('/kaggle/input/currency-excahnge-rate/currency_exchange_rate.csv')
rates.head()
ex_rates.head(3)
rates.tail()
rates.shape
ex_rates.shape
rates.columns
ex_rates.iloc[5:7]
ex_rates[ ex_rates.TIME == 2001]
ex_rates[ ex_rates.LOCATION =='AUS'].Value.mean()
ex_rates[ ex_rates.LOCATION =='JPN'].sort_values('Value', ascending = True)
ex_rates.groupby('LOCATION').Value.mean()
ex_rates.groupby('LOCATION').size()
ex_rates[ ex_rates.LOCATION == 'AUS'].plot.line(x='TIME', y = 'Value')