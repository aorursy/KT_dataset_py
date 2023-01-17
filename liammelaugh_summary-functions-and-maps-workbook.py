import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
print(np.median(reviews.points))
print(reviews.points.median())
check_q1(88.0)
# Your code here
check_q2(reviews.country.unique())
# Your code here
reviews.groupby("country").size()[reviews.groupby("country").size().isin(np.sort(reviews.groupby("country").size())[-5:])]
check_q3(reviews.groupby("country").size()[reviews.groupby("country").size().isin(np.sort(reviews.groupby("country").size())[-5:])])
reviews.country.value_counts()
#This answer is much better than mine but notepad is quite rubbish when it comes to code completion
#I mean how is anyone supposed to know there exists a method value_counts at least in Spyder you would get
#a full list with that in it. It's the main reason I don't like Jupyter notebooks
reviews.price.head()
reviews.price.median()
# Your code here
median = reviews.price.median()
reviews.price.map(reviews.price-median)#This does not work right
#check_q4(reviews.price.map(reviews.price-median))
reviews.price.map(lambda v: v - median)#Much better
check_q4(reviews.price.map(lambda v: v - median))
# Your code here
reviews["points_price_ratio"] = reviews.points/reviews.price
#reviews["points_price_ratio"] = reviews["points_price_ratio"].dropna()
reviews.iloc[reviews.points_price_ratio.idxmax()]
#reviews.points_price_ratio.idxmax()
check_q6(reviews.iloc[reviews.points_price_ratio.idxmax()])#check_q5 is wrong so is 6
#answer_q5()# This answer is way off. It has nothing to do with the question
#Anyway reviews.price.values-median is faster
#reviews.price.values-median
#reviews.price.apply(lambda v: v - median)
#Neither of these answer the question
#reviews["points_price_ratio"].max()
# Your code here
check_q7(pd.Series([sum(["tropical" in val for val in reviews.description]), sum(["fruity" in val for val in reviews.description])], index=("tropical","fruity")))
# Your code here
country_variety_df = reviews.loc[:,["country","variety"]].dropna()
#country_variety_df.head()
country_variety_df["country_variety"] = [country_variety_df.iloc[i].country + "_" + country_variety_df.iloc[i].variety for i in range(len(country_variety_df))]
country_variety_df.country_variety.value_counts()
check_q8(country_variety_df.country_variety.value_counts())#How is this wrong 
country_variety_df["country_variety"] = country_variety_df.apply(lambda df: df.country + "_" + df.variety, axis = "columns")
country_variety_df.country_variety.value_counts()
check_q8(country_variety_df.country_variety.value_counts())# This check is wrong