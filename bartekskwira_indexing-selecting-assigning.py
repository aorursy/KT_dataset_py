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
a1 = reviews.description
check_q1(a1)
# Your code here
a2 = reviews.description[0]
check_q2(a2)
# Your code here
a3 = reviews.iloc[0]
check_q3(a3)
# Your code here
a4 = reviews.description[0:10]
check_q4(a4)
# Your code here
a5 = reviews.iloc[[1, 2, 3, 5, 8]]
check_q5(a5)
# Your code here
a6 = reviews.loc[[0, 1, 10, 100], ["country", "province", "region_1", "region_2"]]
check_q6(a6)
# Your code here
a7 = reviews.loc[0:100, ["country", "variety"]]
check_q7(a7)

# Your code here
a8 = reviews[reviews.country == 'Italy']
check_q8(a8)
# Your code here
a9 = reviews[reviews.region_2.notnull()]
check_q9(a9)
# Your code here
a10 = reviews["points"]
check_q10(a10)

# Your code here
a11 = reviews.loc[:1000, "points"]
check_q11(a11)
# Your code here
a12 = reviews.iloc[-1000:, 3]
check_q12(a12)

# Your code here
a13 = reviews.loc[reviews.country == 'Italy', "points"]
check_q13(a13)
# Your code here
#a14 = reviews[((reviews.country == "France") | (reviews.country == "Italy")) & (reviews.points >= 50)]["country"]
a14 = reviews[reviews.country.isin(["France", "Italy"]) & (reviews.points >= 90)].country
check_q14(a14)
