import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.describe().T
# Your code here
reviews.points.median()
check_q1(reviews.points.median())
# Your code here
reviews.country.unique()
check_q2(reviews.country.unique())
# Your code here
reviews.country.value_counts()
check_q3(reviews.country.value_counts())
# Your code here
reviews_price_median = reviews.price.median()
reviews.price.map(lambda p: p - reviews_price_median)
check_q4(reviews.price.map(lambda p: p - reviews_price_median))
reviews.columns
# Your code here
#i have a wine 10point and £5 = 10:5, 2:1
#i have a wine for 5points and £10 = 5:10, 1:2

#reviews.points / reviews.price #this is applied to all rows
bargin = reviews.loc[(reviews.points / reviews.price).idxmax()].title
bargin
# Your code here
#source: https://codereview.stackexchange.com/questions/189061/extracting-specific-words-from-pandas-dataframe
wine_words = ['tropical', 'fruity']
words = reviews.description.str.split(expand=True).stack() #split all the strings, stack
words = words[words.isin(wine_words)]
words = pd.Series(words.value_counts())
words
#source: https://www.kaggle.com/natasha23/summary-functions-maps
tropical = reviews.description.str.contains('tropical').value_counts()
fruity = reviews.description.str.contains('fruity').value_counts()
#make a new series
scores = pd.Series([tropical[True], fruity[True]], index=['tropical', 'fruity'])
scores
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
check_q7(pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity']))
# Your code here
#remove null rows
reviews.loc[:, ['country', 'variety']].isnull().sum()
new_df = reviews.dropna(subset=['country', 'variety'])
#concat country and variety
new_s = pd.Series(new_df.country + '-' + new_df.variety)
#generate a series counting unique values
new_s.value_counts()
result = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = result.apply(lambda row: row.country + " - " + row.variety, axis=1)#axis='columns' also works
ans.value_counts()
check_q8(ans.value_counts())
answer_q8()
