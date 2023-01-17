############### Imports ######################

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Local module imports 

from m5_helpers import get_rollup, get_w_df



############################### Load data ###########################

prices_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

ss = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

cal_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

train_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')



################## Series for all 12 levels ##################

# We will need an aggregating matrix "fit" on train_df. 

# Good thing we previously made a function to get that. 

rollup_matrix_csr, rollup_index  = get_rollup(train_df)



# We want a dataframe with all the aggregated series. 

series_df = pd.DataFrame(data=rollup_matrix_csr * train_df.iloc[:, 6:].values,

                         index=rollup_index, 

                         columns=train_df.iloc[:, 6:].columns)
################# stats ######################

# Lets if we can use describe and transpose 

# to get some statistical features of different columns

series_df.T.loc[:, (1, slice(None))].describe().T
%%time

############ Create stats_df ################

stats_df = series_df.T.describe().T

stats_df.head()
# This is good, but I'd like to do a bit better. For one thing, count is 

# not really helpful as levels 10-12 because leading zeros indicate that 

# an item is not for sale yet, and therefore should not be included in 

# count. The min column is also not useful because christams will give 

# close to zero sales for all series since Walmart is closed on christmas.

# Finally, I would like to add the relevant percentiles that will be 

# used in the uncertainty competition. 



################### Leading zeros ######################



# We would like to set all leading zeros to np.nan so 

# they won't be counted in by the .describe() method. 

# To do this we need a mask for series_df that only shows

# the leading zeros. If we compare series_df 

# values to cumulative sum values multiplied by any 

# number != 1 (2 chosen here), the values are only equal

# if they are at a location of a leading zero. 

zero_mask = series_df.cumsum(axis=1) * 2 == series_df



# Now set the leading zeros to np.nan

series_df[zero_mask] = np.nan





################## Christmas closure ####################

# First find all x where 'd_x' represents christmas. 

xmas_days = cal_df[cal_df.date.str[-5:] == '12-25'].d.str[2:].astype('int16')



# I will choose to replace sales for every christmas with 

# the average of the day before and the day after. 

for x in xmas_days: 

    series_df[f'd_{x}'] = (series_df[f'd_{x-1}'] + series_df[f'd_{x+1}']) / 2

    

    

################ Percentiles ######################

# These will be especially useful in the uncertainty competition. 

percentiles = [.005, .025, .165, .25, .5, .75, .835, .975, .995]



############### Recreate stats_df #################

stats_df = series_df.T.describe(percentiles).T



################## fraction 0 #######################

# We want to know what fraction of sales are zero 

stats_df['fraction_0'] = ((series_df == 0).sum(axis = 1) / stats_df['count'])



############### Add weights ###################

w_df = get_w_df(train_df, cal_df, prices_df, rollup_index, rollup_matrix_csr, start_test=1914)

stats_df = pd.concat([stats_df, w_df], axis=1)



############### Pickle files ##################

stats_df.to_pickle('stats_df.pkl')

series_df.to_pickle('series_df.pkl')
stats_df.info()
series_df.info()
series_df.head()
## THIS WILL BE PUT IN THE HELPERS.PY FILE AND WILL BE UPDATED THERE, NOT HERE.



####################################### Module ##########################################

#########################################################################################

import pandas as pd

################### series_df function #####################

def get_series_df(train_df, rollup_matrix_csr, rollup_index, cal_df):

    """Returns a dataframe with series for all 12 levels of aggregation. We also 

    replace leading zeros with np.nan and replace christmas sales with average 

    of the day before and day after christmas"""

    

    series_df = pd.DataFrame(data=rollup_matrix_csr * train_df.iloc[:, 6:].values,

                         index=rollup_index, 

                         columns=train_df.iloc[:, 6:].columns)

    

    zero_mask = series_df.cumsum(axis=1) * 2 == series_df



    # Now set the leading zeros to np.nan

    series_df[zero_mask] = np.nan



    ################## Christmas closure ####################

    # First find all x where 'd_x' represents christmas. 

    xmas_days = cal_df[cal_df.date.str[-5:] == '12-25'].d.str[2:].astype('int16')



    # I will choose to replace sales for every christmas with 

    # the average of the day before and the day after. 

    for x in xmas_days: 

        series_df[f'd_{x}'] = (series_df[f'd_{x-1}'] + series_df[f'd_{x+1}']) / 2

    

    return series_df 







################## stats_df function #######################

def get_stats_df(series_df, cal_df):

    """Returns a dataframe that shows basic stats for all 

    series in sereis_df."""

    

    ################ Percentiles ######################

    # These will be especially useful in the uncertainty competition. 

    percentiles = [.005, .025, .165, .25, .5, .75, .835, .975, .995]





    ############# Create stats_df ########################

    stats_df = series_df.T.describe(percentiles).T



    ################## fraction 0 #######################

    # We want to know what fraction of sales are zero 

    stats_df['fraction_0'] = ((series_df == 0).sum(axis = 1) / stats_df['count'])

    

    return stats_df