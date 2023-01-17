import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")
reviews.head()
# Your code here
desc = reviews.loc[:,"description"]
print(type(desc))
print(type(reviews))
print(desc)
q1.check()
# Uncomment the line below to see a solution
#q1.solution()
first_description = reviews.loc[0,'description']

q2.check()
print(first_description)
#q2.solution()
first_row = reviews.iloc[0,:]

q3.check()
print(type(first_row))
#q3.solution()
first_descriptions = reviews.loc[0:9,'description']

q4.check()
first_descriptions
#q4.solution()
sample_reviews = reviews.iloc[[1,2,3,5,8],:]
#or
# sample_reviews = reviews.loc[[1,2,3,5,8],:]

q5.check()
sample_reviews
q5.solution()
indices = [0,1,10,100]
category = ['country','province','region_1','region_2']
df = reviews.loc[indices,category]

q6.check()
df
#q6.solution()
category=['country','variety']
df = reviews.loc[0:99,category]

q7.check()
df
q7.solution()
italian_wines = reviews[reviews.country == 'Italy']

q8.check()
q8.solution()
top_oceania_wines = reviews[(reviews.points >= 95) & ((reviews.country == 'Australia') |
                                                    (reviews.country == 'New Zealand'))]
string = top_oceania_wines.loc[345,'description']
q9.check()
print(string)
q9.solution()