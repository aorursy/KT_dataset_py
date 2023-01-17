import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
one = reviews.points.median()
check_q1(one)
# Your code here
two = reviews.country.unique()
check_q2(two)
# Your code here
three = reviews.country.value_counts()
check_q3(three)
# Your code here
mPrice = reviews.price.median()
four = reviews.price.map(lambda price: price - mPrice)
check_q4(four)
# Your code here
five = reviews.loc[(reviews.points / reviews.price).argmax()].title
five
check_q6(five)
# Your code here

tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()

six = pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])

check_q7(six)
# Your code here
seven = reviews.loc[(reviews.country.isnull() == False) 
                  & (reviews.variety.isnull() == False)]

seven = seven.country + ' - ' + seven.variety
seven = seven.value_counts()
check_q8(seven)