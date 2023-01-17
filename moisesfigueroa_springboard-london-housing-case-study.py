import pandas as pd
import numpy as np
path = '../input/uk-housing-prices/UK_House_price_index.xlsx'
data = pd.read_excel(path, sheet_name='Average price', index_col= None)
data
data.info()
data.dropna(how='all', axis=1, inplace=True)
data.drop(columns=['City of London'],inplace = True)
data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
data = data.loc[1:, 'Date': 'Westminster']
data.columns
data = pd.melt(data, id_vars = 'Date', var_name='District', value_name='Price')
data.Price = data.Price.astype('float')
data.info()
data.District.nunique()
data['year'] = pd.DatetimeIndex(data.Date).year
last_20_yrs = data.loc[data.year >= 2001].copy()
last_20_yrs = last_20_yrs.groupby(['District', 'year']).Price.mean().reset_index().rename(columns={'Price': 'avg_price'})
last_20_yrs
last_20_yrs['pt_change'] = last_20_yrs.groupby('District').avg_price.pct_change(periods=19)
last_20_yrs.loc[last_20_yrs.pt_change == last_20_yrs.pt_change.max(), ['District','pt_change']]
