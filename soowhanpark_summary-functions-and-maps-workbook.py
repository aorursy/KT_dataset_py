import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
a=reviews.points.median()
check_q1(a)
b=reviews.country.unique()
check_q2(b)
c=reviews.country.value_counts()
#c
check_q3(c)
review_price_median=reviews.price.median()
d=reviews.price.map(lambda p: p - review_price_median)
check_q4(d)
median = reviews.price.median()
e=reviews.price.apply(lambda x: x-median)
check_q5(e)
#(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
#check_q7, not check_q6
tropical=reviews.description.map(lambda x: True if "tropical" in x else False)
fruity=reviews.description.map(lambda x: True if "fruity" in x else False)
t=tropical.value_counts()[1]
f=fruity.value_counts()[1]
import pandas as pd
check_q7(pd.Series([t, f], index=['tropical','fruity']))
df=reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
conc = df.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
count=conc.value_counts()
check_q8(count)