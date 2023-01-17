import pandas as pd
pd.set_option('max_rows', 10)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
reviews['points'].median()
# another initial exploratory function is describe()
reviews['country'].unique()
reviews['country'].value_counts()
median_price = reviews['price'].median()
reviews['price'].map(lambda p: p - median_price)
# map takes every value in the column it is being called on and converts it 
# to some new value using a function you provide it
# map takes a Series as input
# For a simple substraction like the above doing the following would yield same result faster
# reviews['price] - median_price
reviews.iloc[(reviews['points'] / reviews['price']).idxmax()].title
# unlike the proposed solution to the current q5 I have substituted
# idxmax() for the deprecated argmax() funtion.
# idxmax() returns the Index of ONLY the first occurrence of maximum of values.
# in this case the value 64590 which is then used by the (i)loc operator
# but in reality there are TWO wines that have a 21.5 ratio and the following
# would be more correct not using the idxmax() function
reviews[(reviews.points/reviews.price) == (reviews.points/reviews.price).max()].title
# the same solution can also be found using this other approach
# by using numpy function nanmax which will Return the maximum of an array
# or maximum along an axis, ignoring any NaNs.
reviews.loc[(reviews.points / reviews.price) == np.nanmax((reviews.points / reviews.price))].title
# this solution is obtained as follows
tropical = reviews.description.map(lambda d: 'tropical' in d).value_counts()
fruity = reviews.description.map(lambda d: 'fruity' in d).value_counts()
# this will map the records having the string 'tropical' or 'fruity' in the description to True
# and the value_counts() function will return the number of 
# False (tropical=126364) (fruity=120881) and True (tropical=3607) (fruity=9090)
# so if I assign the value_count to a variable then tropical[True] will return 
# the number of description in which the word tropical (and fruity) was found
# the following will put the numbers into a Series with proper index names
pd.Series([tropical[True],fruity[True]], index=['tropical', 'fruity'])
# an alternative solution could be
pd.Series([reviews.description.map(lambda p: 'tropical' in p).sum(), 
           reviews.description.map(lambda p: 'fruity' in p).sum()],
            index=['tropical', 'fruity'])
# as an added reminder please note there are 374 Fruity and 204 Tropical strings
# which are NOT counted with the above solutions but here follows total solution
pd.Series([reviews.description.map(lambda p: 'tropical' in p).sum(), 
           reviews.description.map(lambda p: 'Tropical' in p).sum(),
           reviews.description.map(lambda p: 'fruity' in p).sum(),
           reviews.description.map(lambda p: 'Fruity' in p).sum()],
            index=['tropical', 'Tropical', 'fruity', 'Fruity'])

# use the loc function to select only rows in which BOTH the country and the variety
# are not null. In reality variety is never null but let's ignore this
# also select only the country and variety columns to build the new ans dataframe
ans = reviews.loc[(reviews['country'].notnull()) & (reviews['variety'].notnull()),['country','variety']]
# now use the apply function to create a series catenating country a dash and variety
# I am using the formatted print as the catenating funcntion and not the + operator
# since the former is better for readability and performance
# see https://softwareengineering.stackexchange.com/questions/304445/why-is-s-better-than-for-concatenation
ans = ans.apply(lambda srs: "%s - %s" % (srs.country, srs.variety), axis='columns')
# while following is the official suggestion with the + catenation
# ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts().head(10)