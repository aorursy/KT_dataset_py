import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews['description'])

check_q2(reviews['description'][0])
check_q3(reviews.iloc[0])
series = reviews['description'][0:10]
check_q4(series)
df = reviews.iloc[[1,2,3,5,8]]
check_q5(df)
df = reviews[['country', 'province', 'region_1', 'region_2']].iloc[[0,1,10,100]]
print(df)
check_q6(df)
df = reviews.iloc[0:101][['country', 'variety']]
print(df)
check_q7(df)
df = reviews[reviews.country == 'Italy']
check_q8(df)
df = reviews.dropna(subset=['region_2'])
print(df)
check_q9(df)
points =  reviews.points  # must be series; these are being plotted
print(points)
check_q10(points)
points_1000 = reviews.iloc[0:1001].points
check_q11(points_1000)
points_1000 = reviews.iloc[-1000:].points
check_q12(points_1000)
points_italy = reviews[reviews.country == 'Italy'].points
check_q13(points_italy)
points_france_vs_italy = reviews[((reviews.country == 'Italy') | (reviews.country == 'France'))]
over_90 = points_france_vs_italy[points_france_vs_italy.points >= 90].country # answer must be series of country
print(over_90)
check_q14(over_90)
