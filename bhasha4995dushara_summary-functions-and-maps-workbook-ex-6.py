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
reviews_median_price = reviews.price.median()
reviews.price.map(lambda p : p - reviews_median_price)
reviews.iloc[(reviews.points/reviews.price).idxmax()]['title']
tropical_wine = reviews.description.map(lambda r : "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True],fruity_wine[True]],index=['tropical','fruity'])
#df = pd.DataFrame(data=reviews,columns=['country','variety'])
#df = df.dropna()
#g = reviews.country+' - '+reviews.variety
#g.value_counts()        
# or
a = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
a.apply(lambda s:s.country + ' - ' + s.variety,axis=0).value_counts()