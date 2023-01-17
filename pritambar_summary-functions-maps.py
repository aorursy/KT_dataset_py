import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(20)
reviews.head()
reviews.points.median()
print(reviews['country'].unique())
check_q2(reviews['country'].unique())
print(reviews.country.value_counts())
#print(reviews['country'].mode())
med = reviews['price'].median()
reviews.price.map(lambda v: v - med)
reviews.loc[(reviews.points / reviews.price).idxmax()].title
tropical = reviews.description.map(lambda r : "tropical" in r).value_counts()
fruity = reviews.description.map(lambda r : "fruity" in r).value_counts()
print(tropical, fruity)
s1 = pd.Series((tropical[True],fruity[True]), index = ("tropical", "fruity"))
print(s1)
new = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
new = new.apply(lambda s: s.country + " - " + s.variety, axis = 'columns')
print(new)
print(new.value_counts())
check_q7(new.value_counts())