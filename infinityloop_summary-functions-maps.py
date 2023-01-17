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
check_q1(reviews.points.median())
# Your code here
check_q2(set(reviews.country))
# Your code here
reviews.country.value_counts()
check_q3(reviews.country.value_counts())
# Your code here
# reviews.price = reviews.price - reviews.price.median()
answer_q4()
# Your code here
answer = reviews.iloc[(reviews.points / reviews.price).idxmax].title
expected= reviews.loc[(reviews.points / reviews.price).argmax()].title
# answer_q5()
print(answer)
print(expected)
print(answer == expected)
# Your code here
answer_6 = pd.Series({'tropical':reviews.description.str.contains('tropical').sum(),'fruity':reviews.description.str.contains('fruity').sum()})
print(answer_6)
# check_q6()
# answer_q6()
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])

# Your code here
answer = reviews[['country', 'variety']].dropna(axis=0)
answer = answer.apply(lambda x:x.country + ' -  ' + x.variety, axis=1)
answer_count_my = answer.value_counts()
print(answer_count_my)

ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans_count_expected = ans.value_counts()
print(ans_count_expected)

print(answer_count_my.sort_index(inplace=True) == ans_count_expected.sort_index(inplace=True))