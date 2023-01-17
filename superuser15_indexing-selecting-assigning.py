import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews.columns
description = reviews.description
check_q1(description)
# Your code here
val = description[0]
print(val)
check_q2(val)
# Your code here
row1 = reviews.loc[0]
print(type(row1))
check_q3(row1)
# Your code here
des_10 = description[0:10]
print(des_10)
check_q4(des_10)
# Your code here
df = reviews.iloc[[1,2,3,5,8]]
check_q5(df)
print(answer_q5())
# Your code here

cols = ['country','province','region_1','region_2']
#df2 = reviews[cols]
#df3 = df2.iloc[[0,1,10,100]]
df4 = reviews.loc[[0,1,10,100],cols]
check_q6(df4)
#print(answer_q6())
# Your code here
df5 = reviews.loc[0:100,['country','variety']]
# Your code here
df6 = reviews.loc[ reviews.country == 'Italy']

# Your code here
wines = reviews.loc[reviews.region_2.notnull()]

# Your code here
print(reviews.columns)
points = reviews.points
#print(answer_q10())
# Your code here
p = reviews.loc[0:1000,'points']
check_q11(p)
print(p.head())
print(answer_q11())
print(reviews.shape)
# Your code here
r = reviews.iloc[-1000:,3]
# Your code here
points = reviews.points[reviews.country == 'Italy']
#print(points)
check_q13(points)
# Your code here
ans = reviews[reviews.country.isin(["Italy",'France']) & (reviews.points >= 90)].country
ans.head()