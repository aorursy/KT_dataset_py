import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
#reviews.points.describe()
reviews.points.median()
# Your code here
reviews.country.unique()
# Your code here
reviews.country.value_counts()
# Your code here
reviews.points.map(lambda p : p - reviews.points.median());
# Your code here
reviews.loc[(reviews.points / reviews.price).idxmax()].title

# Your code here
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
# Your code here
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts()