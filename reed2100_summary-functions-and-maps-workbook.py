import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews['points'].median()
# Your code here
reviews['country'].unique()
# Your code here
reviews['country'].value_counts()
# Your code here
price_med = reviews['price'].median()
reviews['price'].map(lambda p: p - price_med)
# Your code here

reviews.loc[(reviews['points']/reviews['price']).idxmax()]['title']


# Your code here

tropic = reviews['description'].map(lambda x : 'tropical' in x).value_counts()
fruity = reviews['description'].map(lambda x : 'fruity' in x).value_counts()

pd.Series([tropic[True],fruity[True]], index=['Tropical','Fruity'] )

tropic


# Your code here

#first attempt, this worked and it was in 3 steps, want to see if there's a better way
#in this attempt I didn't use a lambda function, mapping or applying, instead I just used pandas built in understanding of series operations and
#made a new column
#but I know that this takes up space and becomes more data to work with 
valid_country_and_variety = reviews[(reviews['country'].notnull()) & (reviews['variety'].notnull())]

reviews['country_variety'] = valid_country_and_variety['country'] + ' - ' + valid_country_and_variety['variety']

reviews['country_variety'].value_counts()



reviews[reviews['country'].notnull() & reviews['variety'].notnull()].apply(lambda frame : frame['country']+' - '+frame['variety'],axis='columns').value_counts()

#tried doing it all in one line this time (after examining the provided solution and maps a bit)
#first try gave me a big error, I had to specify the axis = columns in the lambda function...need to re-read up on specifying axes

#without doing the value counts at the end, I confirmed that APPLYING the lambda function to the non-null country/var frame returns a series
#guess it's like..."return the result of adding these two columns from the given dataframe argument", and not "add a new col to the frame"
#so a lambda / map function can be flexible...
#this is the provided solution, looks a little cleaner than my first 

#first made the non-null frame
#then apply a lambda function to it, returning the result of concatenating the country / var colums (making sure to go across cols)
#then did a value_counts on the returned series 
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts()