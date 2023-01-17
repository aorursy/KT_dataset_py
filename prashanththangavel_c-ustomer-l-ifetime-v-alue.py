import numpy as np 
import pandas as pd

hist = pd.read_csv('../input/historical_transactions.csv')
hist = hist[['card_id','purchase_date','purchase_amount']]
hist = hist.sort_values(by=['card_id', 'purchase_date'], ascending=[True, True])
hist.head()
## Time
from datetime import datetime

z = hist.groupby('card_id')['purchase_date'].max().reset_index()
q = hist.groupby('card_id')['purchase_date'].min().reset_index()

z.columns = ['card_id', 'Max']
q.columns = ['card_id', 'Min']

## Extracting current timestamp
now = datetime.now()
curr_date = now.strftime("%m-%d-%Y, %H:%M:%S")
curr_date = pd.to_datetime(curr_date)

rec = pd.merge(z,q,how = 'left',on = 'card_id')
rec['Min'] = pd.to_datetime(rec['Min'])
rec['Max'] = pd.to_datetime(rec['Max'])

## Time value 
rec['Recency'] = (curr_date - rec['Max']).astype('timedelta64[D]') ## current date - most recent date

## Recency value
rec['Time'] = (rec['Max'] - rec['Min']).astype('timedelta64[D]') ## Age of customer, MAX - MIN

rec = rec[['card_id','Time','Recency']]
rec.head()
## Frequency
freq = hist.groupby('card_id').size().reset_index()
freq.columns = ['card_id', 'Frequency']
freq.head()
## Monitary
mon = hist.groupby('card_id')['purchase_amount'].sum().reset_index()
mon.columns = ['card_id', 'Monitary']
mon.head()
final = pd.merge(freq,mon,how = 'left', on = 'card_id')
final = pd.merge(final,rec,how = 'left', on = 'card_id')

final['historic_CLV'] = final['Frequency'] * final['Monitary'] 
final['AOV'] = final['Monitary']/final['Frequency'] ## AOV - Average order value (i.e) total_purchase_amt/total_trans
final['Predictive_CLV'] = final['Time']*final['AOV']*final['Monitary']*final['Recency'] 

final.head()