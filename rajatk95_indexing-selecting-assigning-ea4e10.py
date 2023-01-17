import pandas as pd
import seaborn as sns


import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews['description'])
check_q2(reviews['description'].head(1))
check_q3(reviews.iloc[0])
check_q4(reviews.head(10)['description'])
l = [1,2,3,5,8]
check_q5(reviews.iloc[l])
i = [0,1,10,100]
header = ['country', 'province', 'region_1', 'region_2']
check_q6(pd.DataFrame(reviews, index=i, columns=header))
header = ['country', 'variety']
first100 = reviews.loc[0:100, header];
check_q7(first100)
# check_q8()
iscountryItaly = reviews['country'] == "Italy"
check_q8(reviews[iscountryItaly])
check_q9(reviews[reviews['region_2'].notnull()])
check_q10(reviews['points'])
check_q11(reviews['points'].head(1000))
check_q12(reviews['points'].tail(1000))
filtereddf = reviews[reviews["country"] == "Italy"].loc[:,'points']
check_q13(filtereddf)

isFranceItaly = reviews.country.isin(["France", "Italy"])
isPointg8ThanEq90 = reviews.points >=90
filtereddf = reviews[isFranceItaly & isPointg8ThanEq90]
check_q14(filtereddf.country)

