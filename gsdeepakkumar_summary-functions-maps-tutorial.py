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
check_q1(reviews.points.median())
# Your code here
reviews.country.unique()
check_q2(reviews.country.unique())
# Your code here
reviews.country.value_counts()
check_q3(reviews.country.value_counts())
# Your code here
price_median=reviews.price.median()
reviews.price.map(lambda p:p-price_median)
answer_q4()
reviews.loc[(reviews.points / reviews.price).idxmax()].title
# Your code here
#tropical=reviews.description.map()
tropical=reviews.description.map(lambda p:'tropical' in p).value_counts()
fruity=reviews.description.map(lambda p:'fruity' in p).value_counts()
tropical
fruity
wine_type=pd.Series([tropical[True],fruity[True]],index=['tropical','fruity'])
wine_type
answer_q6()
# Your code here
count_winvar=reviews.loc[(reviews.country.notnull() & (reviews.variety.notnull()))]
count_winvar=count_winvar.apply(lambda p:p.country + "-"+ p.variety,axis='columns')
count_winvar.value_counts()
answer_q7()