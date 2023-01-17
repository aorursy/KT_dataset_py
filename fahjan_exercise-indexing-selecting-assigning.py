import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.indexing_selecting_and_assigning import *

print("Setup complete.")
reviews.head()
# Your code here

# desc = reviews['description']

desc = reviews.description



q1.check()

desc.head()

#q1.hint()

#q1.solution()
#first_description = desc[0]

first_description = desc.loc[0]





q2.check()

first_description
#q2.hint()

#q2.solution()
first_row = reviews.loc[0]



q3.check()

first_row
#q3.hint()

#q3.solution()
# first_descriptions = reviews.description[0:10]

first_descriptions = reviews.loc[0:9, "description"]

# first_descriptions = reviews.iloc[0:10, "description"]



q4.check()

first_descriptions
#q4.hint()

#q4.solution()
sample_reviews = reviews.iloc[[1, 2, 3, 5, 8]]



q5.check()

sample_reviews
#q5.hint()

#q5.solution()
# df = reviews.loc[reviews.country == 'Italy', ['country', 'province', 'region_1', 'region_2']]

df = reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]





q6.check()

df
#q6.hint()

#q6.solution()
df = reviews.loc[0:99, ['country', 'variety']]



# df = reviews[['country', 'variety']].loc[0:99]



q7.check()

df
from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())



print (pysqldf("SELECT * FROM reviews LIMIT 2;").head())
#q7.hint()

#q7.solution()
italian_wines = reviews[reviews.country == 'Italy']



q8.check()
#q8.hint()

#q8.solution()
top_oceania_wines = reviews[(reviews.country.isin(['Australia', 'New Zealand'])) & reviews.points > 95]



q9.check()

top_oceania_wines
#q9.hint()

#q9.solution()