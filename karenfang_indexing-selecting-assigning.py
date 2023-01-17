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
reviews['description']
check_q1(reviews['description'])
# Your code here
reviews.description[0]
check_q2(reviews.description[0])
# Your code here
reviews.iloc[0]
check_q3(reviews.iloc[0])
# Your code here
d = reviews.description.iloc[0:10]
df = pd.Series(data = d)
print(df)
check_q4(df)
# Your code here
reviews.iloc[[1,2,3,5,8]]
check_q5(reviews.iloc[[1,2,3,5,8]])
# Your code here
answer = reviews[['country', 'province','region_1','region_2']].iloc[[0,1,10,100]]
answer
check_q6(answer)
# Your code here
reviews[['country','variety']].iloc[0:101]
check_q7(reviews[['country','variety']].iloc[0:101])
# Your code here
reviews[reviews.country == 'Italy']
check_q8(reviews[reviews.country == 'Italy'])
# Your code here
reviews[reviews.region_2.notnull()]
check_q9(reviews[reviews.region_2.notnull()])
# Your code here
reviews.points
check_q10(reviews.points)
# Your code here
reviews.points.iloc[0:1001]
check_q11(reviews.points.iloc[0:1001])
# Your code here
reviews.points.iloc[-1000:]
check_q12(reviews.points.iloc[-1000:])
# Your code here
reviews.points[reviews.country == 'Italy']
check_q13(reviews.points[reviews.country == 'Italy'])
# Your code here
reviews[reviews.points >=90].country[(reviews.country == 'Italy') | (reviews.country == 'France')]
check_q14(reviews[reviews.points >=90].country[(reviews.country == 'Italy') | (reviews.country == 'France')])
