import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
ex1 = reviews['description']
check_q1(ex1)
ex2 = ex1[0]
check_q2(ex2)
# Your code here
ex3 = reviews.iloc[0]
print(check_q3(ex3))
ex3
print(reviews.columns)
ex4 = reviews.iloc[0:10, 1]
check_q4(ex4)
print(ex4) #Notice we get the first 10 values, as its inclusive:exclusive
ex5 = reviews.iloc[[1,2,3,5,8]]
print(check_q5(ex5))
ex5
ex6 = reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
check_q6(ex6)
ex7 = reviews.loc[0:99,['country','variety']]
print(check_q7(ex7))
ex7
ex8 = reviews.loc[reviews.country == 'Italy']
print(check_q8(ex8))
ex8
ex9 = reviews.loc[reviews.region_2.notnull()]
check_q9(ex9)

# Your code here
check_q10(reviews.points)
check_q11(reviews.points.iloc[0:1000])
check_q12(reviews.points.iloc[-1000:])
reviews.columns
check_q13(reviews.points.loc[reviews.country == 'Italy'])
check_q14(reviews.country.loc[((reviews.country == 'Italy') | (reviews.country == 'France')) & (reviews.points >= 90)])