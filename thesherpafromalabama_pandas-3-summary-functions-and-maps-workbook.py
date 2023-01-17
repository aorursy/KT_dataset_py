import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews.points.median()
# Your code here
reviews.country.nunique() #Total number of unique values
reviews.country.value_counts() #Counts instances for each unique value
reviews.country.unique() #Returns an array of unique values

# Your code here
most_often = reviews.country.value_counts()
check_q3(most_often)
# Your code here
mp = reviews.price.median()
remap = reviews.price.map(lambda x: x - mp)
check_q4(remap)
answer_q4()
# Your code here wine = point/price
reviews['n'] = 0
reviews.n.map(lambda x,y: x / y, reviews.points, reviews.price)
reviews.max()

# Your code here

# Your code here
