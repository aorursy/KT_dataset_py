import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()

reviews.points.median()
# Your code here
check_q1(reviews.points.median())
# Your code here
a2= reviews.country.unique()
check_q2(a2)
# Your code here
a3 = reviews.country.value_counts()
check_q3(a3)
# Your code here
offset = reviews.price.median()
a4 = reviews.price.map(lambda p: p - offset)
check_q4(a4)
answer_q4()
# Your code here 

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

offset = reviews.price.median()
print(offset)
def remedianprice(srs):
    srs.price = srs.price - offset
    return srs

print(reviews.price.mean())
reviews = reviews.apply(remedianprice, axis = 'columns')
 
print(reviews.price.mean())
print(check_q4(reviews)) 
reviews
# Your code here

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

reviews['bargain'] = reviews.apply(lambda x: x.points/x.price, axis = 1)
print(reviews.bargain)
 
    
a6 = reviews.iloc[reviews['bargain'].argmax()]
check_q5(a6.title)
tropicalCount = sum(reviews.description.str.contains('tropical'))
fruityCount = sum(reviews.description.str.contains('fruity'))
a7= pd.Series([tropicalCount, fruityCount],index = ['tropical','fruity'])
a7
check_q6(a7)
# Your code here
reviews = reviews[reviews.variety.notnull() & reviews.country.notnull()]
myseries = reviews.country + ' - ' + reviews.variety
check_q7(myseries.value_counts())