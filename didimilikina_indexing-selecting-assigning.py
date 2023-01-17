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

check_q1(reviews['description'])
# Your code here

check_q2(reviews['description'].head(1))
# Your code here

check_q3(reviews.iloc[0])
# Your code here

check_q4(reviews.head(10)['description'])
# Your code here

l = [1,2,3,5,8]

check_q5(reviews.iloc[l])
# Your code here

i = [0,1,10,100]

header = ['country', 'province', 'region_1', 'region_2']

check_q6(pd.DataFrame(reviews, index=i, columns=header))
# Your code here

header = ['country', 'variety']

first100 = reviews.loc[0:100, header];

check_q7(first100)
# Your code here

iscountryItaly = reviews['country'] == "Italy"

check_q8(reviews[iscountryItaly])
# Your code here

check_q9(reviews[reviews['region_2'].notnull()])
# Your code here

check_q10(reviews['points'])
# Your code here

check_q11(reviews['points'].head(1000))
# Your code here

check_q12(reviews['points'].tail(1000))
# Your code here

filtereddf = reviews[reviews["country"] == "Italy"].loc[:,'points']

check_q13(filtereddf)
# Your code here

isFranceItaly = reviews.country.isin(["France", "Italy"])

isPointg8ThanEq90 = reviews.points >=90

filtereddf = reviews[isFranceItaly & isPointg8ThanEq90]

check_q14(filtereddf.country)