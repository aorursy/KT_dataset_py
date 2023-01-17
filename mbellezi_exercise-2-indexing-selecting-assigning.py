import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")
reviews.head()
desc = reviews.description
#desc = reviews["description"]
desc
type(desc)
first_description = reviews.description.iloc[0]
#first_description = reviews.description.loc[0]
first_description
first_row = reviews.iloc[0]
first_row
first_descriptions = reviews.loc[:9, 'description']
#first_descriptions = reviews.description.iloc[:10]
#first_descriptions = reviews.description.head(10)
first_descriptions
sample_reviews = reviews.loc[[1, 2, 3, 5, 8]]
#indexes = [1, 2, 3, 5, 8]
#sample_reviews = reviews.iloc[indexes]
sample_reviews
cols = ['country', 'province', 'region_1', 'region_2']
indexes = [0, 1, 10, 100]
df = reviews.loc[indexes, cols]
df
df = reviews.loc[:99, ['country', 'variety']]
df = reviews.iloc[:100, [0, 11]]
df
italian_wines = reviews.loc[reviews.country == 'Italy']
#italian_wines = reviews.loc[(reviews.country == 'Italy') & (reviews.price <= 16.0)]
italian_wines
top_oceania_wines = reviews.loc[
    ((reviews.country == "Australia") | (reviews.country == "New Zealand"))
    & (reviews.points >=95)]
top_oceania_wines