import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
check_q1(reviews['description'])
# Your code here
check_q2(reviews['description'].iloc[0])
# Your code here
check_q3(reviews.iloc[0])
# Your code here
check_q4(reviews.description.head(10))
# Your code here
check_q5(reviews.filter([1,2,3,5,8], axis=0))
# Your code here
check_q6(reviews[["country","province","region_1","region_2"]].filter([0,1,10,100],axis=0))
# Your code here
check_q7(reviews[["country","variety"]].iloc[0:101])
# Your code here
countryfilter =  reviews.country == 'Italy'
#countryfilter
check_q8(reviews[countryfilter])
# Your code here
region_2_nonNan = reviews.region_2.notna()
check_q9(reviews[region_2_nonNan])
# Your code here
check_q10(reviews['points'])
# Your code here
check_q11(reviews['points'].iloc[0:1000])
# Your code here
check_q12(reviews['points'].iloc[-1000:])
# Your code here
check_q13(reviews[reviews.country == 'Italy']['points'])
# Your code here
pointsfilter = reviews.points >= 90
italyfilter = reviews.country   == 'Italy'
francefilter = reviews.country == 'France'
check_q14(reviews[pointsfilter & (italyfilter | francefilter) ] [['points','country']])