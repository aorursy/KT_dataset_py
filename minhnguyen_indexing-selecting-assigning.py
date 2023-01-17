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
check_q1(reviews["description"])
# Your code here
check_q2(reviews["description"][0])
# Your code here
check_q3(reviews.iloc[0])
reviews.iloc[0]

# Your code here
answer_q4()
#Select the first 10 values from the description column of reviews
check_q4(reviews.loc[0:9, 'description'])


check_q5(reviews.iloc[[1, 2, 3, 5, 8], :])
# Your code here
check_q6(reviews.loc[[0, 1, 10, 100], ["country", "province", "region_1", "region_2"]])
# Your code here
check_q7(reviews.loc[0:100, ["country", "variety"]])
reviews.loc[0:100, ["country", "variety"]]

# Your code here
check_q8(reviews.loc[reviews.country == "Italy"])
# Your code here
check_q9(reviews.loc[reviews.region_2.notnull()])
# Your code here
check_q10(reviews.loc[:, "points"])
# Your code here
check_q11(reviews.loc[0:1000, "points"])
# Your code here
check_q12(reviews.loc[-1000:, "points"])
check_q12(reviews[-1000:].points)
# Your code here
check_q13(reviews.loc[reviews.country == "Italy"].points)
# Your code here
check_q14(reviews.loc[reviews.country.isin(['Italy', 'France']) &(reviews.points >= 90)].country)