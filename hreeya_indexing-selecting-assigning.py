import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
a=reviews['description']
check_q1(a)
# Your code here
b=reviews['description'][0]
check_q2(b)
# Your code here
c=reviews.iloc[0]
check_q3(c)
# Your code here
d=reviews.iloc[:10,1]
check_q4(d)
# Your code here
e=reviews.loc[[1,2,3,5,8],:]
check_q5(e)

# Your code here
f=reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
check_q6(f)
# Your code here
g=reviews.loc[0:99,['country','variety']]
check_q7(g)
# Your code here
h=reviews.loc[reviews.country=='Italy']
check_q8(h)

# Your code here
i=reviews.loc[reviews.region_2.notnull()]
check_q9(i)
# Your code here
j=reviews.points
check_q10(j)

# Your code here
k=reviews.loc[0:999,'points']
check_q11(k)

# Your code here
l=reviews.iloc[-1000: ,3]
check_q12(l)

# Your code here
m=reviews.loc[reviews.country=='Italy','points']
check_q13(m)
# Your code here
contr=reviews[reviews.country.isin(['Italy','France'])&(reviews.points>=90)].country
check_q14(contr)