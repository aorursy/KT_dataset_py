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
print(reviews['description'])
check_q1(reviews['description'])
# Your code here
check_q2(reviews['description'][0])
# Your code here
check_q3(reviews.iloc[0,:])

# Your code here
check_q4(reviews.iloc[0:10,1])

# Your code here
check_q5(reviews.loc[[1,2,3,5,8],:])
# Your code here
check_q6(reviews.loc[[0,1,10,100],['country','province','region_1','region_2']])
# Your code here
check_q7(reviews.loc[0:100,['country','variety']])

# Your code here
check_q8(reviews.loc[reviews.country == 'Italy'])
# Your code here
check_q9(reviews.loc[reviews.region_2.notna()])
# Your code here
reviews.points
# Your code here
reviews.points[0:1000]
# Your code here
reviews.points[-1000:]
# Your code here
reviews[reviews.country == "Italy"].points
# Your code here
reviews[reviews.country.isin(["France", "Italy"]) & (reviews.points >= 90)].country
