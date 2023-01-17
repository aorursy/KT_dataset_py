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
reviews.description
# Your code here
reviews.description[0]
# Your code here
reviews.loc[0, :]
# Your code here
reviews.description.iloc[0:10]
# Your code here
reviews.iloc[[1,2,3, 5, 8], :]
# Your code here
reviews.loc[[0,1,10, 100], ['country', 'province', 'region_1', 'region_2']]
# Your code here
reviews.loc[0:100, ['country', 'variety']]
# Your code here
reviews.loc[reviews.country == 'Italy']
# Your code here
reviews.loc[reviews.region_2.notnull()]
# Your code here
check_q10(reviews.points)
# Your code here
check_q11(reviews.points.iloc[0:1001])
# Your code here
check_q12(reviews.points.iloc[-1000:])
# Your code here
check_q13(reviews.loc[reviews.country == 'Italy'].points)
# Your code here
check_q14(reviews.loc[ reviews.country.isin(['Italy', 'France']) &(reviews.points >= 90)].country)