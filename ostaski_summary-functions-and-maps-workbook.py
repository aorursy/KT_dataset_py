import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()
# 88.0
reviews['country'].unique()
# array(['Italy', 'Portugal', 'US', 'Spain', 'France', 'Germany',
#       'Argentina', 'Chile', 'Australia', 'Austria', 'South Africa',
#       'New Zealand', 'Israel', 'Hungary', 'Greece', 'Romania', 'Mexico',
#       'Canada', nan, 'Turkey', 'Czech Republic', 'Slovenia',
#       'Luxembourg', 'Croatia', 'Georgia', 'Uruguay', 'England',
#       'Lebanon', 'Serbia', 'Brazil', 'Moldova', 'Morocco', 'Peru',
#       'India', 'Bulgaria', 'Cyprus', 'Armenia', 'Switzerland',
#       'Bosnia and Herzegovina', 'Ukraine', 'Slovakia', 'Macedonia',
#       'China', 'Egypt'], dtype=object)
reviews.country.value_counts()
# US          54504
# France      22093
#            ...  
# Egypt           1
# Slovakia        1
# Name: country, Length: 43, dtype: int64
reviews['price']
# 0          NaN
#1         15.0
#          ... 
#129969    32.0
#129970    21.0
#Name: price, Length: 129971, dtype: float64

reviews['price'].map(lambda x: x - reviews.price.median())

reviews['price']
# 0          NaN
# 1        -10.0
#          ... 
# 129969     7.0
# 129970    -4.0
#Name: price, Length: 129971, dtype: float64
reviews.price
#0          NaN
#1         15.0
#          ... 
#129969    32.0
#129970    21.0
#Name: price, Length: 129971, dtype: float64

reviews.price.apply(lambda x: x - reviews.price.median())

reviews.price
# 0          NaN
# 1        -10.0
#          ... 
# 129969     7.0
# 129970    -4.0
reviews.title[(reviews.points / reviews.price).idxmax()]
# 'Bandit NV Merlot (California)'
tropical = reviews.description.map(lambda d: "tropical" in d).value_counts()
fruity = reviews.description.map(lambda d: "fruity" in d).value_counts()

pd.Series([tropical[True], fruity[True]], index=['tropical', 'fruity'])
# tropical    3607
# fruity      9090
# dtype: int64
result = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
result.apply(lambda row: row.country + " - " + row.variety, axis='columns')
# 0               Italy - White Blend
# 1         Portugal - Portuguese Red
#                    ...            
# 129969          France - Pinot Gris
# 129970      France - Gewürztraminer
# Length: 129907, dtype: object