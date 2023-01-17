import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
q1 = reviews.points.median()
check_q1(q1)
# Your code here
q2 = reviews.country.unique()
check_q2(q2)
# Your code here
q3 = reviews.country.mode()
q3
q3 = reviews.country.value_counts()
check_q3(q3)
# Your code here
med = reviews.price.median()
q4 = reviews.price.map(lambda p: p - med)
med
check_q4(q4)
reviews.price.head()
# Your code here
def sub_med(serie):
    serie.price = serie.price - med
    return serie
q5 = reviews.apply(sub_med, axis='columns')
q5.head()
check_q5(q5)
answer_q5()
# Your code here
q6 = reviews.points / reviews.price
ans = reviews.iloc[q6.idxmax()]
check_q6(ans)
# Your code here
tropical = reviews.description.map(lambda d: 'tropical' in d)
cnt_trop = tropical.value_counts()
fruity = reviews.description.map(lambda d: 'fruity' in d)
cnt_fruit = fruity.value_counts()
answer = pd.Series([cnt_trop[True], cnt_fruit[True]], index=['tropical', 'fruity'])
check_q7(answer)
# Your code here
country_variety = reviews.country + " - " + reviews.variety
country_variety = country_variety.dropna()
q8 = country_variety.value_counts()
check_q8(q8)