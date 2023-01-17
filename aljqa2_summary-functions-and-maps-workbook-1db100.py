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
review_price_mean = reviews.price.mean()
#reviews.loc[(reviews.points / reviews.price).argmax()].title
reviews.loc[(reviews.points /reviews.price).idxmax()].title
#reviews.iloc[64590].title
reviews[(reviews.points/reviews.price) == (reviews.points/reviews.price).max()].title
reviews.iloc[[64590,126096], :]
#review_points_mean = reviews.points.mean()
#reviews.points.map(lambda p: p - review_points_mean)
#tropical_count = reviews[reviews.description.str.contains('tropical')].title.count()
#fruity_count = reviews[reviews.description.str.contains('fruity')].title.count()
#print(type(fruity_count))
#answer_q6()

tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
d = reviews.loc[reviews.country.notnull() & reviews.variety.notnull()]
d = d.apply(lambda srs: srs.country + ' ' + srs.variety, axis= 'columns')
d.value_counts()