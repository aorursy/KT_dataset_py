import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")
reviews.head()
# Your code here
desc = reviews.loc[:,'description']

q1.check()


# Uncomment the line below to see a solution
q1.solution()

reviews.head()
first_description = reviews.loc[0,'description']

q2.check()
first_description
q2.solution()
first_row = reviews.iloc[0,:]

q3.check()
first_row
q3.solution()
first_descriptions = ____

q4.check()
first_descriptions
#q4.solution()
indices = [1,2,3,5,8]
sample_reviews = reviews.loc[indices]

sample_reviews
q5.solution()
df = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
q6.check()

q6.solution()
df = reviews.loc[0:99, ['country','variety']]

q7.check()
df
q7.solution()
italian_wines = reviews[reviews.country == 'Italy']
q8.check()
q8.solution()
top_oceania_wines = reviews[(reviews.points >=95) & (reviews.country.isin(['Australia','New Zealand']))]

q9.check()
q9.solution()