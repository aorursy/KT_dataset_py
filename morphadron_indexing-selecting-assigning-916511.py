import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(reviews.head())
check_q1(reviews.description)
reviews.description
# Your code here
ans = reviews.description[0]
check_q2(ans)
# Your code here
reviews.iloc[0]
# Your code here
df1=reviews.iloc[:,1]
answer = df1[0:10]
check_q4(answer)
df1 = reviews.iloc[[1,2,3,5,8],:]
check_q5(df1)
# Your code here
df1 = reviews.iloc[[0,1,10,100],[0,5,6,7]]
df1
check_q6(df1)
# Your code here
df1=reviews.iloc[:,[0,11]]
df2=df1[0:101:1]
df2

# Your code here
ans = reviews.loc[reviews.country == 'Italy']
check_q8(ans)
ans
ans = reviews.loc[reviews.region_2.notnull()]
#check_q9(ans)
ans
points = reviews.points
points = reviews.points
first_1000= points[0:1001:1]# Your code here
first_1000
points= reviews.points
last_1000 = points[:-1001:-1]
last_1000
# Your code here
df1 = reviews.iloc[:,[0,3]]
df2 = df1.loc[df1.country == 'Italy']
df2
df1 = reviews.loc[((reviews.country == 'Italy') | (reviews.country == 'France'))  & (reviews.points >= 90)]
df2 = df1.iloc[:,[0,3]]
df2