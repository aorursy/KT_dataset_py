import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())

check_q1(reviews.description)
# Your code here
check_q2(reviews.iloc[0].description)
# Your code here
check_q3(reviews.iloc[0])
check_q4(reviews.iloc[:10].description)
# Your code here
reviews.iloc[[1,2,3,5,8]]
# Your code here
reviews.loc[[0,1,10,100],["country","province","region_1","region_2"]]
# Your code here
check_q7(reviews.loc[0:99,["country","variety"]])
# Your code here
check_q8(reviews[reviews.country=="Italy"])
# Your code here
check_q9(reviews[reviews.region_2.notna()])
# Your code here
check_q10(reviews.points)
# Your code here
check_q11(reviews.points.iloc[:1000])
# Your code here
# Your code here
check_q13(reviews[reviews.country=="Italy"].points)
# Your code here
answer_q14()