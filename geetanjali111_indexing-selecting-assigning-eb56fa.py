import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews["description"]
check_q1(reviews["description"])
reviews["description"][0]
reviews.description[0]
reviews.iloc[0]
reviews.description.iloc[0:10]
reviews.loc[[1,2,3,5,8],:]
reviews.loc[[0,1,10,100],["country","province","region_1","region_2"]]
reviews.loc[0:100,["country","variety"]]
reviews.loc[reviews.country=="Italy"]
check_q8(reviews.loc[reviews.country=="Italy"])
reviews.loc[reviews.region_2.notnull()]
check_q9(reviews.loc[reviews.region_2.notnull()])
reviews.points
check_q10(reviews.points)
data=reviews.winery.isnull()
reviews.points[reviews.winery[0:1000]]
check_q11(reviews.points[reviews.winery[0:1000]])
reviews.points
answer_q11(0)
answer_q11()
reviews.loc[:999, 'points']
reviews.points[-1000:]
reviews.points[reviews.country=="Italy"]
check_q13(reviews.points[reviews.country=="Italy"])
reviews[(reviews.country.isin(["France","Italy"])) & (reviews.points>=90)].country
#answer_q14()