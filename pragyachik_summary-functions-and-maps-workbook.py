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
rpm=reviews.price.median()
check_q4(reviews.price.map(lambda pri:pri-rpm)) 
# Your code here
#reviews_points=reviews.points/reviews.price
#reviews['bestbuy']=reviews.apply(lambda row: row.points/row.price, axis=1)
#reviews['bestbuy'].idxmax()
#points_per_price = reviews.points / reviews.price
#check_q5(print(points_per_price.idxmax()))
#print(reviews.loc[reviews.bestbuy == reviews.bestbuy.idxmax(),'country'])
reviews.loc[(reviews.points / reviews.price).idxmax()].title
#print(answer_q6())
# Your code here
answer_q7()
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
# Your code here
answer_q8()
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts()