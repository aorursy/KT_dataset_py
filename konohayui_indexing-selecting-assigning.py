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
reviews.description[0]
reviews.iloc[0]
reviews.iloc[:10, 1]
reviews.iloc[[1, 2, 3, 5, 8]]
reviews[["country", "province", "region_1", "region_2"]].iloc[[0, 1, 10, 100]]
reviews[["country", "variety"]].iloc[:100]
reviews[reviews.country == "Italy"]
reviews[reviews.region_2.notnull()]
data = reviews.points
sns.countplot(data)
sns.countplot(reviews.points[:1000])
sns.countplot(reviews.points[-1000:])
sns.countplot(reviews[reviews.country == "Italy"].points)
reviews[reviews["country"].isin(["France", "Italy"]) & (reviews.points >= 90)]["country"]