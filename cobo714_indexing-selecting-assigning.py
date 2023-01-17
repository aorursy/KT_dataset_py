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
reviews["description"][0]
reviews.iloc[0]
reviews.description[:10]
reviews.iloc[[1,2,3,5,8]]
reviews.iloc[[0,1,10,100],].loc[:,["country","province","region_1","region_2"]]
reviews.loc[0:100, ["country", "variety"]]
reviews[reviews.country=="Italy"]
reviews[reviews.region_2.notnull()]
reviews.points
check_q10(reviews.points)
reviews.points[:1000]
check_q11(reviews.points[:1000])
reviews.points[-1000:]
check_q12(reviews.points[-1000:])
reviews.points[reviews.country == "Italy"]
check_q13(reviews.points[reviews.country == "Italy"])
reviews[reviews.country.isin(["France", "Italy"]) & reviews.points >= 90].country
check_q14(reviews[reviews.country.isin(["France", "Italy"]) & (reviews.points >= 90)].country)