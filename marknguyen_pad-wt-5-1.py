import pandas as pd

pd.options.display.max_rows = 15
# Import clean transaction data
df = pd.read_excel('../input/transactions_clean.xlsx')
## Calculate the mean, and median of the quantity_sold and mean of the total_net_amount by date and item_name

## Calculate total dollars sold for each day


## Calculate total dollars sold for each day by product


## Calculate sales mix for each product

## Create a summary table using pivot_table for values: quantity_sold and total_amount. The row should be date and the columns should be the product names

df.pivot_table(values=['quantity_sold','total_amount'],
               index=['date'],
               columns=['item_name'],
               aggfunc=['sum'],
               fill_value='-')