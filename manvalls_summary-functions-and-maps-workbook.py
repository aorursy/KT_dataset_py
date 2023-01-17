import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
check_q1(reviews.points.median())
# Your code here
check_q2(reviews.country.unique())
# Your code here
check_q3(reviews.country.value_counts())
# Your code here
median = reviews.price.median()
check_q4(reviews.price.map(lambda p: p - median))
# Your code here
check_q6(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
# Your code here
fruitty = reviews.description.map(lambda desc: 'fruity' in desc)
tropical = reviews.description.map(lambda desc: 'tropical' in desc)
check_q7(pd.Series([tropical[tropical == True].size, fruitty[fruitty == True].size], index=['tropical', 'fruity']))
# Your code here
nona = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
variations = nona.country + ' - ' + nona.variety
check_q8(variations.value_counts())
