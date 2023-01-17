import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
check_q1(reviews.description)
# Your code here
check_q2(reviews.description.head(1))
# Your code here
check_q3(reviews.loc[0])
# Your code here
check_q4(reviews.description.head(10))
# Your code here
check_q5(reviews.iloc[[1, 2, 3, 5, 8]])
# Your code here
check_q6(reviews.loc[[0, 1, 10, 100], ["country", "province", "region_1", "region_2"]])
# Your code here
check_q7(reviews.loc[0:99, ["country", "variety"]])
# Your code here
check_q8(reviews[reviews["country"] == "Italy"])
# Your code here
check_q9(reviews[reviews["region_2"].isna() == False])
# Your code here
check_q10(reviews.points)
# Your code here
check_q11(reviews.loc[:999, "points"])
# Your code here
check_q12(reviews.iloc[-1000:, 3])
# Your code here
check_q13(reviews[reviews["country"] == "Italy"]["points"])
# Your code here
check_q14(reviews[reviews["country"].isin(["Italy", "France"]) & (reviews.points >= 90)].country)
