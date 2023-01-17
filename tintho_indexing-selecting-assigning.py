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
desc = reviews['description']
print(check_q1(desc))

desc
# Your code here
firstDesc = desc[0]
print(check_q2(firstDesc))

firstDesc
# Your code here
print(check_q3(reviews.iloc[0]))
reviews.iloc[0]
# Your code here
q4 = desc.iloc[range(10)]
print(check_q4(q4))
q4
# Your code here
i = [1,2,3,5,8]
print(check_q5(reviews.iloc[i]))

reviews.iloc[i]
# Your code here
cols = ['country', 'province', 'region_1', 'region_2']
print(check_q6(reviews[cols].iloc[[0,1,10,100]]))
reviews[cols].iloc[[0,1,10,100]]
# Your code here
cols = ['country', 'variety']
print(check_q7(reviews[cols].loc[0:100]))

reviews[cols].loc[0:100]
# Your code here

print(check_q8(reviews.loc[reviews.country == 'Italy']))
reviews.loc[reviews.country == 'Italy']
# Your code here
# print(check_q9())

print(check_q9(reviews.loc[pd.notnull(reviews.region_2)]))
reviews.loc[pd.notnull(reviews.region_2)]
# Your code here
print(check_q10(reviews['points']))
reviews['points']
# Your code here
check_q11(reviews['points'].iloc[0:1000])
reviews['points'].iloc[0:1000]
# Your code here
print(check_q12(reviews['points'].iloc[-1000:]))
reviews['points'].iloc[-1000:]
# Your code here
check_q13(reviews['points'].loc[reviews.country == 'Italy'])
reviews['points'].loc[reviews.country == 'Italy']
# Your code here
italy = reviews['country'] == 'Italy'
france = reviews['country'] == 'France'

check_q14(reviews['country'][italy | france].loc[reviews.points >= 90])
reviews['country'][italy | france].loc[reviews.points >= 90]
# reviews[italy & france]
# reviews['country'].loc[(reviews.country == 'Italy' or reviews.country =='France' )] #| reviews.country == 'France') & reviews.points >= 90]