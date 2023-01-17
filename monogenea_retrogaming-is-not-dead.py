import numpy as np

import pandas as pd

import re

import seaborn as sns

from matplotlib import pyplot as plt



# Select relevant columns and rename

sales = pd.read_csv('../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv',

                   usecols = ['Name','Platform','Year_of_Release','Genre','Publisher','NA_Sales'])

sales.columns = ['name', 'system', 'year', 'genre', 'publisher', 'sales']



# Drop NAs

sales.dropna(inplace = True)

# Drop null sales

sales.query('sales > 0', inplace = True)

# Re-encode 'year' as int32

sales.year = sales.year.astype('int32')



# Log-transform sales and compare distributions

sales['log_sales'] = np.log(sales.sales + 1)

sales[['sales', 'log_sales']].hist(figsize = (12, 4), bins = 50)
# Select Nintendo systems

nintendo = ['NES','SNES','GB','GC','N64','GBA']

sales = sales.query('system in @nintendo')



# Group by year of release and platform (MultiIndex) to compute sum of log NA sales accordingly

group_sales = sales.groupby(['year', 'system']).sum().log_sales



# Unstack group_sales and set NAs to zero

tidy = group_sales.unstack().fillna(0)

tidy.plot(figsize = (12, 6))
pd.crosstab(sales.year, columns = 'number games').plot.bar(y = 'number games', figsize = (12, 6))
# Group by year of release and platform (MultiIndex) to compute median of log NA sales accordingly

group_sales = sales.groupby(['year', 'system']).median().log_sales



# Unstack group_sales and set NAs to zero

tidy = group_sales.unstack().fillna(0)

tidy.plot(figsize = (12, 6))
NUM_PUB = 5 # number of publishers to be displayed



genreByPub = pd.crosstab(sales.publisher, sales.genre)

topFive = genreByPub.sum(axis = 1).sort_values(ascending = False).index[:NUM_PUB]



# Horizontal barplot

genreByPub.loc[topFive].plot.barh(figsize = (8, 12), grid = True, legend = 'reverse')