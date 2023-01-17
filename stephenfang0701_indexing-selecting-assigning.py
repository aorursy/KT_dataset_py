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
reviews.description # or reviews["description"]
# Your code here
reviews.description[0]
# reviews["description"].iloc[0]
# Your code here
# Use loc for label index, and use iloc for position index
reviews.iloc[0]
# Your code here
reviews.iloc[:10]["description"]
# reviews.iloc[:10, 1]
# Your code here
reviews.iloc[[1,2,3,5,8]]
# Your code here
reviews.loc[[0, 1, 10, 100], ["country", "province", "region_1", "region_2"]]
# reviews.loc[[0, 1, 10, 100]][["country", "province", "region_1", "region_2"]]
print(answer_q6())
# Your code here
reviews.loc[0:100, ["country", "variety"]]
# reviews.loc[0:100][["country", "variety"]]
reviews.country=="Italy"
# Your code here
reviews.loc[reviews.country=="Italy"]
reviews.region_2.notnull()
# Your code here
reviews.loc[reviews.region_2.notnull()]
# Your code here
check_q11(reviews.points)
sns.distplot(reviews.points, bins=20, kde=False)
# Your code here
check_q11(reviews.loc[0:1000, "points"])
reviews.iloc[-1000:]
# Your code here
check_q12(reviews.iloc[-1000:]["points"])
# Your code here
check_q13(reviews.loc[reviews.country=="Italy", "points"])
# Your code here
check_q14(reviews[(reviews.country.isin(["France", "Italy"])) & (reviews.points >= 90)].country)