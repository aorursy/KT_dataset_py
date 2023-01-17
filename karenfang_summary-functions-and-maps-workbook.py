import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews.points.median()
check_q1(reviews.points.median())
# Your code here
reviews.country.unique()
check_q2(reviews.country.unique())
# Your code here
reviews.country.value_counts()
check_q3(reviews.country.value_counts())
# Your code here
medprice = reviews.price.median()
reviews.price.map(lambda p: p - medprice)
check_q4(reviews.price.map(lambda p: p - medprice))
# Your code here
medprice = reviews.price.median()
reviews.price.apply(lambda p: p - medprice)
check_q5(reviews.price.apply(lambda p: p - medprice))
# Your code here
points_to_price = reviews.points/reviews.price
reviews.title[(points_to_price).idxmax()]
check_q6(reviews.title[(points_to_price).idxmax()])
tropical = reviews.description.map(lambda w: "tropical" in w).value_counts()
fruity = reviews.description.map(lambda w: "fruity" in w).value_counts()
pd.Series([tropical[True],fruity[True]],index=['tropical','fruity'])
# Your code here
check_q7(pd.Series([tropical[True],fruity[True]],index=['tropical','fruity']))
# Your code here
df = reviews[['country','variety']][(reviews.country.notnull()) & (reviews.variety.notnull())]

comb = df.apply(lambda r: r.country + ' - ' + r.variety, axis = 'columns' )

comb.value_counts()
check_q8(comb.value_counts())