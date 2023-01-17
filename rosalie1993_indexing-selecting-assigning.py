import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
a=reviews.description
a
check_q1(a)
# Your code here
a=reviews.description[0]
a
check_q2(a)
# Your code here
a=reviews.iloc[0]
a
check_q3(a)
# Your code here
b=reviews.loc[:9,'description']
c = pd.Series(b)
c
check_q4(c)
# Your code here
a = reviews.loc[[1,2,3,5,8]]
a
check_q5(a)
# Your code here
a = reviews.loc[[0,1,10,100],['country', 'province', 'region_1', 'region_2']]
a
check_q6(a)
# Your code here
a = reviews.loc[0:100,['country' , 'variety']]
a
check_q7(a)
# Your code here
a = reviews.country == 'Italy'
b = reviews.loc[a]
b
check_q8(b)

# Your code here
a = reviews.region_2 == 'NaN'
b = reviews.loc[reviews.region_2.notnull()]
b
#check_q9(b)
# Your code here
a = reviews.loc[:,'points']
#a
b = reviews.points
#b
#print(a == b)
check_q10(b)
#check_q10(a)
#print(answer_q10())
# Your code here
a = reviews.loc[0:1000,'points']
a
check_q11(a)
# Your code here
a = reviews.iloc[-1000:,3]
a
check_q12(a)
# Your code here
a = reviews.country == 'Italy'
b = reviews.loc[a,'points']
b
check_q13(b)
# Your code here
a = reviews.country == ('France' or 'Italy')
b = reviews.loc[((reviews.country == 'France') | (reviews.country == 'Italy')) & (reviews.points >= 90),'country']
#((reviews.country == 'France') | (reviews.country == 'Italy')) & (reviews.points >= 90)
#print(answer_q14())
#b
#check_q14(b)
reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country