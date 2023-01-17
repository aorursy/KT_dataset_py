import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews ['description'])
check_q2(reviews['description'][0])
check_q3(reviews.iloc [0])
check_q4(reviews.iloc[0:10, 1])
check_q5 (reviews.iloc [[1,2,3,5,8]])
check_q6 (reviews.loc [[0,1,10,100]][['country','province','region_1','region_2']])
check_q7 (reviews.loc[0:100, ['country', 'variety']])
# 0:100 is a mistake on the workbook author's part and the correct answer should be 0:99 for the first 100 records
check_q8 (reviews[reviews['country'] == 'Italy'])
check_q9 (reviews [reviews.region_2.notnull()])
reviews['points']
reviews [:1000]['points']
reviews [-1000:]['points']
Italia = reviews ['country'] == 'Italy'
reviews [Italia] ['points']
Francia = reviews ['country'] == "France"
above_averagely = reviews ['points'] >= 90
reviews [(Francia | Italia) & above_averagely] ['country']