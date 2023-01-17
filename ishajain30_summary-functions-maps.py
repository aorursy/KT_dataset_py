import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(20)
reviews.head()
# Your code here
reviews.points.median()
# Your code here
reviews.country.unique()
# Your code here
reviews.country.value_counts().head()
# Your code here
medprice = reviews.price.median()
reviews.price.map(lambda v: v - medprice)
# Your code here
reviews.loc[(reviews.points/reviews.price).idxmax()].title
# Your code here
findtro = reviews.description.map(lambda v: 'tropical' in v).value_counts()
findfru = reviews.description.map(lambda r: 'fruity' in r).value_counts()
frutro = pd.Series([findfru[True], findtro[True]], index = ['fruity', 'tropical'])
frutro
# Your code here
ds = reviews.loc[(reviews.country.notnull())& (reviews.variety.notnull())]
ds = ds.apply(lambda v: v.country+ " - "+ v.variety, axis = 1)
ds.value_counts()
