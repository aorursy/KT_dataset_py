import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
q = reviews.ix[:,'description']
check_q1(q)
print(q)
# Your code here
w = reviews.ix[0,'description']
check_q2(w)
print(w)
# Your code here
e = reviews.ix[0,:]
check_q3(e)
print(e)
# Your code here
r = reviews.ix[:,'description']
w = r.head(10)
check_q4(w)
print(w)
# Your code here
r = reviews.iloc[[1,2,3,5,8],:]
check_q5(r)
print(r)
# Your code here
t = reviews.iloc[[0,1,10,100],:]
y = t.loc[:,['country','province','region_1','region_2']]
check_q6(y)
print(y)
# Your code here
u = reviews.loc[0:100]
i = u.loc[:,['country','variety']]
print(i)
check_q7(i)
# Your code here
o = reviews.loc[reviews.country.isin(['Italy'])]
print(o)
check_q8(o)
# Your code here
a = reviews.loc[-reviews.region_2.isnull()]
print(a)
check_q9(a)
# Your code here
a = reviews.loc[:,'points']
check_q10(a)
# Your code here
a = reviews.loc[:,'points']
e = a.loc[0:1000]
check_q11(e)
# Your code here
q = reviews.loc[len(reviews)-1000:, 'points']
check_q12(q)
# Your code here
r = reviews.loc[reviews.country.isin(['Italy']), 'points']
print(r)
check_q13(r)
# Your code here
import pandas as pd
t = reviews.loc[reviews.country.isin(['France','Italy']) & (reviews.points >= 90)]
check_q14(t)