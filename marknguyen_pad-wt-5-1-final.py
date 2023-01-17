import pandas as pd

pd.options.display.max_rows = 15
# Import clean transaction data
df = pd.read_excel('../input/transactions_clean.xlsx')
## Calculate the mean, and median of the quantity_sold and mean of the total_net_amount by date and item_name
g = df.groupby(['date','item_name'])
g.agg({'quantity_sold':['mean','median'],
       'total_net_amount':'mean'})
## Calculate total dollars sold for each day
totals_df = df.groupby('date').agg({'total_amount':'sum'})
display(totals_df.head())

## Calculate total dollars sold for each day by product
sums_df = df.groupby(['date','item_name']).agg({'total_amount':'sum'})
display(sums_df.head())

## Calculate sales mix for each product
(sums_df/totals_df).head()
df.pivot_table(values=['quantity_sold','total_amount'],
               index=['date'],
               columns=['item_name'],
               aggfunc=['sum'],
               fill_value='-')