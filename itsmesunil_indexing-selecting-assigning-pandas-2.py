import pandas as pd

import seaborn as sns

from learntools.advanced_pandas.indexing_selecting_assigning import *



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here

reviews.description
# Your code here

reviews['description'].head(1)
# Your code here

reviews.loc[0]
# Your code here

reviews.description.head(10)
# Your code here

reviews.loc[[1,2,3,5,8]]
# Your code here

reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
# Your code here

reviews.iloc[0:101,[0,11]]
# Your code here

reviews[reviews.country=='Italy']

#reviews[reviews['country']=='Italy']['variety']
# Your code here

reviews.loc[reviews.region_2.notnull()]
# Your code here

reviews.points
# Your code here

reviews.points.head(1000)
# Your code here

reviews.points.tail(1000)
# Your code here

reviews[reviews['country']=='Italy']['points']
# Your code here

reviews[reviews.country.isin(['France','Italy']) & (reviews['points'] >= 90)].country