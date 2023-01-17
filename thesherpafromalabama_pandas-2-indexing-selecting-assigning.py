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
reviews.description
check_q1(reviews.description)
# Your code here
reviews.description[0]
check_q2(reviews.description[0])
# Your code here
reviews.iloc[0,:] #can use : to grab from beginning to entered value
check_q3(reviews.iloc[0,:])
# Your code here

desc = reviews['description'][:10] #use loc if trying to get a named column
check_q4(desc)
# Your code here
grab=[1,2,3,5,8]
recs = reviews.loc[grab,:]
check_q5(recs)
# Your code here
grabx=[0,1,10,100]
graby=['country','province','region_1','region_2']
dF=reviews.loc[grabx,graby]
print(dF)
check_q6(dF)
# Your code here
grabx=list(range(0,100))
graby=['country','variety']

candv = reviews.loc[grabx, graby]
candv.shape
check_q7(candv)
#answer_q7()
# Your code here

italianwine = reviews.loc[reviews.country=='Italy']
print(italianwine)

check_q8(italianwine)
# Your code here
validwine=reviews.loc[reviews.region_2.notnull()]


#validwine.equals(validwinereal)
check_q9(validwine)
#print(validwine)
#answer_q9()
# Your code here
points = reviews.loc[:,'points']
check_q10(points)

# Your code here

check_q11(reviews.loc[0:1000,'points'])
# Your code here
reviews.tail(1001).loc[:,'points']
check_q12(reviews.tail(1000).loc[:,'points'])
# Your code here
check_q13(reviews.loc[reviews.country=="Italy"].loc[:,"points"])
#check_q11(reviews.loc[,'points'])
# Your code here
from statistics import mean
#grab = ["France","Italy"]
#.loc[reviews.country == ("France" | "Italy")]
recs = reviews.loc[grab,:]
ans = reviews.loc[reviews.points >= mean(reviews.points)].loc[:,["country"]]
#check_q14(reviews.loc[reviews.points >= mean(reviews.points)].loc[reviews.country == ("France" or "Italy")].loc[:,["country"]])
reviews.country.unique()