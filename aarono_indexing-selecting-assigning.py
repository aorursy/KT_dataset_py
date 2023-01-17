import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here

reviews['description']


# Your code here

reviews['description'][0]


# Your code here

reviews.iloc[0]


# Your code here

reviews['description'].iloc[0:10]

check_q4(reviews['description'].iloc[0:10])
# Your code here

reviews.iloc[[1,2,3,5,8], ]


# Your code here

reviews.iloc[[0,1,10,100], [0,5,6,7]]

# Your code here

reviews[['country', 'variety']][0:101]


# Your code here

reviews[reviews.country == 'Italy']


# Your code here

reviews[reviews.region_2.notnull()]


# Your code here

reviews['points']

check_q10(reviews['points'])
# Your code here

reviews['points'][0:1001]

check_q11(reviews['points'][0:1001])
# Your code here

reviews['points'].tail(1000)

check_q12(reviews['points'].tail(1000))
# Your code here

reviews['points'][reviews.country == 'Italy']

check_q13(reviews['points'][reviews.country == 'Italy'])
# Your code here

#reviews.isin({'country':['Italy', 'France']})

reviews[(((reviews.country == 'Italy') | (reviews.country == 'France')) & (reviews.points >= 90 ))]

check_q14(reviews.country[(((reviews.country == 'Italy') | (reviews.country == 'France')) & (reviews.points >= 90 ))])