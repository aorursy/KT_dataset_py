import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
# Your code here
# Your code here
# Your code here
median = reviews.price.median()
reviews.price.map(lambda p:p - median)
check_q4(reviews.price.map(lambda p:p - median))
# Your code here
ratio = reviews.points / reviews.price
a = ratio.idxmax()
show = reviews.iloc[a]

check_q6(show['title'])


# Your code here

T_number = reviews.description.map(lambda p:'tropical' in p)
F_number = reviews.description.map(lambda p:'fruity' in p)
T_sum = T_number.sum()
F_sum = F_number.sum()
c=pd.Series([T_sum,F_sum],index=['tropical','fruity'])
check_q7(c)
# Your code he
c = reviews.country.loc[reviews.country.notnull()]
v = reviews.variety.loc[reviews.variety.notnull()]
one = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull()),['country','variety']]
def jia(rs):
    return rs.country + ' - ' + rs.variety
two = one.apply(jia,axis='columns')

check_q8(two.value_counts())
