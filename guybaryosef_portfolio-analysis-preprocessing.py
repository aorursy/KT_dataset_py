import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
daily_data = pd.read_csv("../input/48_Industry_Portfolios_daily.CSV", low_memory=False)
monthly_data = pd.read_csv("../input/48_Industry_Portfolios_Wout_Div.csv")

daily_data.head()
daily_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
tmp = daily_data.index[daily_data['Date'] == '19260701'].tolist()
equally_weighted_daily_data = daily_data.loc[24267:]

equally_weighted_daily_data.set_index('Date', inplace=True) # set the date as the index
equally_weighted_daily_data.head()
def split_into_years(original_data, beg_year = 2000, amount_years = 17):
    list_of_new_datasets = []
    for i in range(amount_years):
        # due to weekends, need to make sure we start with correct index
        start_date = (beg_year+i)*(10**4) + 101
        while (str(start_date) not in original_data.index):
            start_date += 1

        end_date = (beg_year+i)*(10**4) + 1231
        while (str(end_date) not in original_data.index):
            end_date -= 1

        list_of_new_datasets.append(original_data.loc[str(start_date):str(end_date)])
    return list_of_new_datasets
daily_return_by_year = split_into_years(equally_weighted_daily_data) # list of our new datasets 
for i, data in enumerate(daily_return_by_year):
    print('Dataset of:', i+2000, end=' --- ')
    for col in data:
        count = (data[col] == ' -99.99').sum() + (data[col] == ' -999').sum() + data[col].isnull().sum()
        if count > 0:
            print(col + ':', count, end="; ")
    print()
for i, data in enumerate(daily_return_by_year):
    data.to_csv('48_IP_eq_w_daily_returns_' + str(i+2000) + '.csv')
sp_daily = pd.read_csv("../input/dailySP500.CSV")
sp_daily.head()
print('Values to impute in Open:', sp_daily['Open'].isnull().sum())
print('Values to impute in Adj Close:', sp_daily['Adj Close'].isnull().sum())
temp_sp_ID =  [ i.replace('-', '') for i in sp_daily['Date'] ]
temp_sp_returns = 100 * (sp_daily['Adj Close'] - sp_daily['Open'] ) / sp_daily['Open']

sp_returns = pd.DataFrame({'Date':  temp_sp_ID, 'Return': temp_sp_returns })
sp_returns.set_index('Date', inplace=True)

sp_returns_by_year = split_into_years(sp_returns)

for i, data in enumerate(sp_returns_by_year):
    data.to_csv('SP_daily_returns_' + str(i+2000) + '.csv')
libor_daily = pd.read_csv("../input/LIBOR USD.csv")
libor_daily.head()
temp_id = []
for i in libor_daily['Date']:
    temp = i.split('.')
    temp_id.append(temp[2] + temp[1] + temp[0])

libor_daily['Date'] = temp_id
libor_daily.head()
print('Values to impute in 3M column:', libor_daily['3M'].isnull().sum())
m = 62

temp_daily_rate = [100*( (1 + i/100)**(1/m) -1) for i in libor_daily['3M'] ]

libor_intr = pd.DataFrame({ 'Date': libor_daily['Date'], 'Effective Daily Interest': temp_daily_rate })
libor_intr.set_index('Date', inplace=True)

libor_intr_by_year = split_into_years(libor_intr)

for i, data in enumerate(libor_intr_by_year):
    data.to_csv('LIBOR_daily_interest_' + str(i+2000) + '.csv')