import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.indexing_selecting_and_assigning import *

print("Setup complete.")
reviews.head()
# Your code here

desc = reviews.description



q1.check()
first_description = reviews.description[0]



q2.check()

first_description
first_row = reviews.iloc[0]



q3.check()

first_row
first_descriptions = reviews.description.iloc[:10]

q4.check()

first_descriptions
sample_reviews = reviews.loc[[1,2,3,5,8]]



q5.check()

sample_reviews
df = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]

q6.check()

df
df = reviews.loc[:99,['country','variety']]



q7.check()

df
italian_wines = reviews[reviews.country == 'Italy']

q8.check()
top_oceania_wines = reviews.loc[(reviews.country.isin(['Australia', 'New Zealand'])) & (reviews.points >= 95)]

q9.check()

top_oceania_wines