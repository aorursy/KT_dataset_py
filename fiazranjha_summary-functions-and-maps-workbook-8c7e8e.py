import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews.points.median()
# Your code here
reviews.country.unique()
# Your code here
reviews.country.value_counts()
# Your code here
print(answer_q4())
median_price=reviews.price.median()
reviews.price.map(lambda p: p-median_price)
# Your code here
reviews.loc[(reviews.points/reviews.price).idxmax()]
#reviews.points/reviews.price
# Your code here
t=reviews.description.map(lambda t: 'tropical' in t).value_counts()
f=reviews.description.map(lambda f: 'fruity' in f).value_counts()
pd.Series([t[True],f[True]],index=['tropical','fruity'])
# Your code here
print(answer_q7())
ans=reviews.loc[(reviews.country.notnull())&(reviews.variety.notnull())]
ans=ans.apply(lambda srs:srs.country + "-"+ srs.variety,axis='columns')
ans.value_counts()