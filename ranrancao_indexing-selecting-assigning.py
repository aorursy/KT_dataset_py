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
#print(answer_q1())
reviews.description
# Your code here
#s = reviews.description[0]
#check_q2(s)
reviews.description[0]
# Your code here
#print(answer_q3())
reviews.iloc[0]
# Your code here
#reviews.description[0:10]
reviews.iloc[0:10, 1]
#print(answer_q4())
#Select the first 10 values from the description column of reviews
# Your code here
#print(answer_q5())
reviews.iloc[[1, 2, 3, 5, 8]]
# Your code here
reviews.iloc[[0, 1, 10, 100], [0,5,6,7]]
#check_q6(s)
#print(answer_q6())

#iloc 是用number做index，loc是用其他做index，很重要区别！！
# Your code here
reviews.loc[0:100, ['country', 'variety']]
#print(answer_q7())
# Your code here
#print(answer_q8())
reviews.loc[reviews.country == 'Italy']
# Your code here
#reviews.loc[reviews.region_2 != "NaN"] 这个是不对的！ 下边的才是对的！表示非空是.notnull()
reviews.loc[reviews.region_2.notnull()]
#print(answer_q9())
# Your code here
#print(answer_q10())
reviews.points
# Your code here
reviews.points.iloc[0:1000]
# Your code here
#reviews.points.iloc[:]
#print(answer_q12())
reviews.points.iloc[-1000:]
# Your code here
#s = reviews.points.loc[reviews.country == "Italy"]
#check_q13(s)
#print(answer_q13())
reviews[reviews.country == "Italy"].points
# Your code here
#print(answer_q14())
s = reviews.country.loc[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)]
check_q14(s)