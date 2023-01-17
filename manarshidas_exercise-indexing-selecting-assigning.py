import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.indexing_selecting_and_assigning import *

print("Setup complete.")
reviews.head()
# Your code here

desc = reviews['description']  ## Or we can use reviews.description

type(desc)
first_description = reviews.description.iloc[0]

first_description
first_row = reviews.iloc[0]

first_row
first_descriptions = reviews.description.iloc[:10]

first_descriptions
indices = [1, 2, 3, 5, 8]

sample_reviews = reviews.loc[indices]

sample_reviews
cols = ['country', 'province', 'region_1', 'region_2']

indices = [0, 1, 10, 100]

df = reviews.loc[indices, cols]

df
cols = ['country', 'variety']

df = reviews.loc[:99, cols]

df
italian_wines = reviews[reviews.country == 'Italy']

italian_wines
top_oceania_wines = reviews.loc[

    (reviews.country.isin(['Australia', 'New Zealand']))

    & (reviews.points >= 95)

]

top_oceania_wines