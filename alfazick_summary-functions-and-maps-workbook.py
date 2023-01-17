import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
ans1 = reviews.points.median()
ans1
check_q1(ans1)
# Your code here
ans2 = reviews.country.unique()
ans2
check_q2(ans2)
# Your code here
ans3 = reviews.country.value_counts()
ans3
check_q3(ans3)
# Your code here
price_median = reviews.price.median()
ans4 = reviews.price.map(lambda p: p - price_median)
check_q4(ans4)
import warnings
warnings.filterwarnings('ignore')
# Your code here
def sub_mean(srs):
    srs.price = srs.price - price_median
    return srs


reviews.apply(sub_mean, axis = 'columns')
reviews.price - price_median
check_q5(reviews.price)
# Your code here
reviews['ratio'] = reviews.points / reviews.price

best_index = np.argmax(reviews.ratio)
ans6 = reviews.loc[best_index] 

check_q5(ans6.title)
# Your code here
tropical = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity = reviews.description.map(lambda r: "fruity" in r).value_counts()
ans7 = pd.Series([tropical[True], fruity[True]], index=['tropical', 'fruity'])
check_q6(ans7)
ans8 = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans8 = ans8.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans8 = ans8.value_counts()
check_q7(ans8)