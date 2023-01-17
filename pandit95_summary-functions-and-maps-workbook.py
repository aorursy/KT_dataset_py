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
#reviews.country.value_counts().idxmax()
reviews.groupby('country').size().idxmax()
# Your code here
reviews_price_median=reviews.price.median()
reviews.price.map(lambda p: p -reviews_price_median)
# Your code here
reviews.loc[(reviews.points/reviews.price).idxmax()].title

# Your code here
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
a=pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
check_q6(a)
# Your code here
df = reviews.xs(['country', 'variety'], axis=1).dropna()
a=df.apply(lambda r: r['country'] + ' - ' + r['variety'], axis=1)
check_q7(a.value_counts())

