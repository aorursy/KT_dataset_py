# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



RAW_DATA = "../input"

hist = pd.read_csv(f'{RAW_DATA}/historical_transactions.csv', low_memory=False, 

                     parse_dates=["purchase_date"])

new =  pd.read_csv(f'{RAW_DATA}/new_merchant_transactions.csv', low_memory=False, 

                     parse_dates=["purchase_date"])



train = pd.read_csv(f'{RAW_DATA}/train.csv', low_memory=False, 

                     parse_dates=["first_active_month"])



test = pd.read_csv(f'{RAW_DATA}/test.csv', low_memory=False, 

                     parse_dates=["first_active_month"])
hist.purchase_date.min(), hist.purchase_date.max(),new.purchase_date.min(), new.purchase_date.max()
from dateutil import rrule

len(list(rrule.rrule(rrule.MONTHLY, dtstart=hist.purchase_date.min(), until=hist.purchase_date.max())))
len(list(rrule.rrule(rrule.MONTHLY, dtstart=new.purchase_date.min(), until=new.purchase_date.max())))
len(list(rrule.rrule(rrule.MONTHLY, dtstart=new.purchase_date.min(), until=hist.purchase_date.max())))
hist['month']= [x.strftime("%y-%m") for x in hist.purchase_date]

counts = hist.month.value_counts().sort_index()

counts.plot(kind='bar')
new['month']= [x.strftime("%y-%m") for x in new.purchase_date]

counts = new.month.value_counts().sort_index()

counts.plot(kind='bar')
hist.month_lag.min(), hist.month_lag.max() , new.month_lag.min(), new.month_lag.max()
tmp = new.loc[new.month_lag == 1, 'purchase_date'].value_counts() < 100

new.loc[tmp.values, 'purchase_date'].min(), new.loc[tmp.values, 'purchase_date'].max()
pd.DataFrame(new.loc[new.month_lag == 1, 'month'].value_counts()).sort_index().plot(

    title="Number of Transactions with month_lag == 1")
new.groupby(['month', 'month_lag']).agg({'card_id': pd.Series.nunique }).T
cnts = new.groupby([ 'month_lag', 'month']).agg({'card_id': pd.Series.nunique, 'purchase_date' : 'count'})

cnts.columns=['number of cards', 'transaction count']
cnts.loc[1,:]
cnts.loc[1,:].plot(title = "monthly counts for month_lag=1")
new_card_ids = new.groupby([ 'month_lag', 'month']).agg({'card_id': list })

hist_cards_monthly_purchase_count = hist.groupby(['card_id', 'month']).agg({'purchase_date': 'count'})
hist_cards_monthly_purchase_count.head(15).T
prom_cards = list(set(new_card_ids.loc[1].loc['17-03'].values[0]))

hist_card_monthly_purchase_count.loc[prom_cards].T.head()
prom_cards = list(set(new_card_ids.loc[1].loc['17-07'].values[0]))

hist_card_monthly_purchase_count.loc[prom_cards].T.head()