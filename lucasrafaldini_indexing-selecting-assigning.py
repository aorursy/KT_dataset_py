import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
df = reviews.description
print(df.head())
check_q1(df)
df = reviews.description[0]
print(df)
check_q2(df)
df = reviews.iloc[0]
print(df)
check_q3(df)
df = reviews.description[:10]
print(df)
check_q4(df)
df = reviews.loc[[1,2,3,5,8],:]
print(df)
check_q5(df)
df = reviews.loc[[0,1,10,100], ['country','province','region_1', 'region_2']]
print(df)
check_q6(df)
df = reviews.loc[:99,['country', 'variety']]
print(df.head())
check_q7(df)
df = reviews.loc[reviews.country=='Italy']
print(df.head())
check_q8(df)
df = reviews.loc[reviews.region_2.notnull()]
print(df.head())
check_q9(df)
df = reviews.points
print(df.head())
print('\n'+'The mean value of points is '+ str(round(df.mean(),2)))
check_q10(df)
df = reviews.points[:100]
print(df.head())
print('\n'+'The mean value of points[:100] is '+ str(round(df.mean(),2)))
check_q11(df)
df = reviews.iloc[-1000:,3]
print(df.head())
print('\n'+'The mean value of the last 1000 elements of column \'points\' is '+ str(round(df.mean(),2)))
check_q12(df)
df = reviews.points[reviews.country=='Italy']
print(df.head())
print('\n'+'The mean value of column \'points\' for Italy is '+ str(round(df.mean(),2)))
check_q13(df)
df = reviews.country[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)]
difference = (reviews.points[(reviews.points >= 90) | (reviews.country == "Italy")].mean()) - (reviews.points[(reviews.points >= 90) | (reviews.country == "France")].mean())
print(df.head())
print('\n'+'The difference between France and Italy mean values of column \'points\', filtering only the best wines with more than 90 points (or equal to),  is '+ str(round(difference,5)))
check_q14(df)