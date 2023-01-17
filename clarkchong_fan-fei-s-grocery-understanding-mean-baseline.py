import itertools



import pandas as pd

import numpy as np
# usecols: select the columns to include, drop the column named id

# dtype: specify the type of the data for certain series

# converters: can define a lambda function, some unit_sales is negative, so convert that to 0 if negative

# nrows: specify the number of rows to read in

# skiprows: skip over this range of rows

df_train = pd.read_csv(

    '../input/train.csv', usecols=[1, 2, 3, 4, 5], dtype={'onpromotion': str},

    converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0},

    skiprows=range(1, 124035460)

)

# nobs = 1461580+1
df_train.tail()
# log transform

df_train["unit_sales"] = df_train["unit_sales"].apply(np.log1p)
# Fill gaps in dates - for example a product has no report in certain store for Jan 2018, but has report

#    for other store in Jan 2018, an entry will be created for that product-store for Jan 2018

#    with value np.nan (NaN)

# Improved with the suggestion from Paulo Pinto

u_dates = df_train.date.unique()

u_stores = df_train.store_nbr.unique()

u_items = df_train.item_nbr.unique()

df_train.set_index(["date", "store_nbr", "item_nbr"], inplace=True)

df_train = df_train.reindex(

    pd.MultiIndex.from_product(

        (u_dates, u_stores, u_items),

        names=["date", "store_nbr", "item_nbr"]

    )

)
# now instead of the default row id, the rows are now labeled by date, store_nbr and item_nbr

df_train.head()
# Fill NAs

df_train.loc[:, "unit_sales"].fillna(0, inplace=True)

# Assume missing entris imply no promotion

df_train.loc[:, "onpromotion"].fillna("False", inplace=True)

df_train.head()
# Calculate means 

df_mean = df_train.groupby(

    ['item_nbr', 'store_nbr', 'onpromotion']

)['unit_sales'].mean().to_frame('unit_sales')

df_mean.head()

# we see that now the date has disappeared, the new df only has one column,

# with index specified in the groupby
# Inverse transform

df_mean["unit_sales"] = df_mean["unit_sales"].apply(np.expm1)

# this creates a master mean reference table that is date-free, now can use that to

# propagate the submission



# Create submission

pd.read_csv(

    "../input/test.csv", usecols=[0, 2, 3, 4], dtype={'onpromotion': str}

).set_index(

    ['item_nbr', 'store_nbr', 'onpromotion']

).join(

    df_mean, how='left'

).fillna(0).to_csv(

    'mean.csv.gz', float_format='%.2f', index=None, compression="gzip"

)