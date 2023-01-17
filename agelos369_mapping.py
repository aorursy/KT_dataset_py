import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()
reviews.country.unique()
reviews.country.value_counts()
price_median = reviews.price.median()
reviews.price.map(lambda p: p - price_median)
i = (reviews.points/reviews.price).argmax()
reviews.loc[i]
sum1 = np.sum(reviews.description.map(lambda p: 'tropical' in p))
sum2 = np.sum(reviews.description.map(lambda p: 'fruity' in p))
s = pd.Series([sum1, sum2], index=['Tropical','Fruity'])
s
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts()