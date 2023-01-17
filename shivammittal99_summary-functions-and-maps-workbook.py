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
median_price = reviews.price.median()
reviews.price.map(lambda val: val - median_price)
# Your code here
reviews.loc[(reviews.points / reviews.price).idxmax()].title
# Your code here
tropical_wine = reviews.description.map(lambda w: 'tropical' in w).value_counts()
fruity_wine = reviews.description.map(lambda w: 'fruity' in w).value_counts()
pd.Series({'tropical': tropical_wine[True], 'fruity': fruity_wine[True]})
# Your code here
df = reviews[reviews.country.notnull() & reviews.variety.notnull()].loc[:, ['country', 'variety']]
df = df.apply(lambda wine: wine.country + ' - ' + wine.variety, axis='columns')
df.value_counts()