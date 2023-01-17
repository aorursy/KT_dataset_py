import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 50)
reviews.head()
check_q1(pd.DataFrame())
reviews.description
answer_q1()
reviews.description[0]
answer_q2()
reviews.iloc[0]
answer_q3()
reviews.iloc[0:10, 1]
answer_q4()
reviews.loc[[1, 2, 3, 5, 8], ['country', 'description', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'taster_name', 'taster_twitter_handle']]

answer_q5()
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
answer_q6()
reviews.loc[0:100, ['country', 'variety']]

answer_q7()
reviews.loc[(reviews.country == 'Italy')]
answer_q8()
reviews.loc[reviews.region_2.notnull()]
answer_q9()
reviews.loc[0:, ['points']]

answer_q10()
reviews.loc[0:1000, ['points']]

answer_q11()
reviews.iloc[-1000:, 3]

answer_q12()
reviews[reviews.country == 'Italy'].points
answer_q13()
reviews.loc[reviews.country.isin(['Italy', 'France']) & (reviews.points >= 90)]
answer_q14()