import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
a =  reviews['description']
print(a)
check_q1(a)
b = reviews.description[0]
print(b)
check_q2(b)
c = reviews.iloc[0,:]
print(c)
check_q3(c)
d = reviews.description.head(10)
print(d)
check_q4(d)
e = reviews.loc[[1,2,3,5,8] , :]
print(e)
check_q5(e)
f = reviews.loc[ [0,1,10,100] , ['country' , 'province' , 'region_1' , 'region_2'] ]
print(f)
check_q6(f)
g = reviews.loc[:100 , ['country' , 'variety']]
print(g)
check_q7(g)
g = reviews.loc[reviews.country == "Italy"]
print(g)
check_q8(g)

i = reviews.loc[reviews['region_2'].notnull()]
print(i)
check_q9(i)
j = reviews['points']
print(j)
print(check_q10(j))
k = reviews['points'].head(1000)
print(k)
print(check_q11(k))
l = reviews['points'].tail(1000)
print(l)
print(check_q12(l))
m = reviews[reviews.country == "Italy"].points
print(m)
print(check_q13(m))

n = reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country
print(n)
check_q14(n)
