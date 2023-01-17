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
df= reviews.description
print(df.head())
print(check_q1(df))

# Your code here
df= reviews.description[0]
print(df)
print(check_q2(df))

# Your code here
df= reviews.iloc[0]
print(df)
print(check_q3(df))

# Your code here
df = pd.Series(reviews.loc[:9, "description"])
print(df)
print(check_q4(df))

# Your code here
df = reviews.loc[[1, 2, 3, 5, 8]]
print(df)
print(check_q5(df))

# Your code here
df= reviews.loc[[0,1,10,100],["country","province","region_1","region_2"]]
print(df)
print(check_q6(df))

# Your code here
df = reviews.loc[:100, ["country","variety"]]
print (df)
print(check_q7(df))

# Your code here
df = reviews[reviews.country == "Italy"]
print(df)
print(check_q8(df))

# Your code here
df = reviews[reviews.region_2.notnull()]
print (df)
print (check_q9(df))
# Your code here
df= reviews.points
print(df)
print (check_q10(df))
# Your code here
df= reviews.points[:1000]
print(df)
print (check_q11(df))
# Your code here
df= reviews.points[-1000:]
print(df)
print (check_q12(df))
# Your code here
df= reviews[reviews.country == "Italy"].points
print(df)
print (check_q13(df))
# Your code here
df = reviews[(reviews.country.isin(["France","Italy"])) & (reviews["points"] >=90)].country
print(df)
print(check_q14(df))