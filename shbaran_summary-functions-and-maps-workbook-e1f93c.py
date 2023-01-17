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
reviews.country
# Your code here
reviews.country.value_counts()
# Your code here
reviews.price.map(lambda p : p - reviews.price.median())
# Your code here
reviews.points / reviews.price
reviews.points.idxmax(skipna = True)
#reviews.points.argmax(skipna = True)
# Your code here
count = {'fruity':0 , 'tropical':0}
for i in range(reviews.shape[0]):
    if 'fruity' in reviews.description[i]:
        count['fruity'] += 1
    if 'tropical' in reviews.description[i]:
        count['tropical'] += 1
print (count)
# Your code here
#print (reviews.country + " " + reviews.variety)
reviews.country.dropna()
reviews.country.value_counts()