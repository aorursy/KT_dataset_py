import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.indexing_selecting_and_assigning import *

print("Setup complete.")
reviews.head()
# Your code here

desc = reviews['description']

type(desc)

q1.check()
#q1.hint()

#q1.solution()
first_description = reviews['description'][0]

first_description = desc[0]



q2.check()

first_description
#q2.hint()

#q2.solution()
first_row = reviews.iloc[0]



# look at documentation or search in google

# how to access first row of pandas dataframe

# this is the right way of doing things, then you remember them over time. 



q3.check()

first_row
#q3.hint()

#q3.solution()
first_descriptions = desc[0:10]



# alternative: first_descriptions = reviews.description.iloc[:10]





q4.check()

first_descriptions
#q4.hint()

#q4.solution()
#indexes = [1,2,3,5,8] 



def fibonacci(num, m):

    seq = [num]

    i = 0

    while m > 0:

        num = num + seq[i-1]

        seq.append(num)

        m -= 1 

        i += 1

    return seq



indexes = fibonacci(1,4)

sample_reviews = reviews.iloc[indexes]



q5.check()

sample_reviews
#q5.hint()

#q5.solution()
columns = ['country', 'province', 'region_1', 'region_2']

indexes = [0, 1, 10, 100]

df = reviews.loc[indexes, columns]



q6.check()

df
#q6.hint()

#q6.solution()
columns = ['country', 'variety']

df = reviews.loc[:99, columns]

#df = reviews.iloc[:100, columns]



q7.check()

df
#q7.hint()

#q7.solution()
italian_wines = reviews.loc[reviews.country == 'Italy']



q8.check()
#q8.hint()

#q8.solution()
oceania_wines = reviews.loc[

    (reviews.country.isin(['Australia','New Zealand']))

    ]

top_oceania_wines = oceania_wines.loc[oceania_wines.points >= 95]



#or all together

#top_oceania_wines = reviews.loc[

#    (reviews.country.isin(['Australia','New Zealand']))

#    & (reviews.points >= 95)

#    ]

q9.check()

top_oceania_wines
q9.hint()

q9.solution()