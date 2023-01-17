# importing library

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



#reading the csv file into a dataframe

ted = pd.read_csv('../input/ted-dataset/ted.csv')



# checking the 1st 5 rows of the data 

ted.head()
# checking the shape of the data (rows, columns)

ted.shape
# cheacking data types

ted.dtypes
# checking missing values

ted.isna().sum()
ted.sort_values('comments').tail()
# which talk have more comments per views

ted['comments_per_views'] = ted.comments / ted.views

ted.sort_values('comments_per_views').tail()
# which talk have les views per comment

ted['views_per_comments'] = ted.views / ted.comments

ted.sort_values('views_per_comments').head()
ted.comments.plot(kind='hist')
# filtering talks with more than or equal to 1000 comments shows that only 32 talks out of 2500

ted[ted.comments >= 1000].shape
# make sense to exclude talks with 1000 comments or more from the plot to get more readiable and informative plot

ted[ted.comments < 1000].comments.plot(kind='hist')
# the code line below does the same thing as the cell code above but with using query method

ted.query('comments < 1000').comments.plot(kind='hist')
# another way is using loc which is more flixable + increasing the number of bis in the histogram to see more details (default bins is 10)

ted.loc[ted.comments < 1000, 'comments'].plot(kind='hist', bins=20)
# take 10 random samples of the event column

ted.event.sample(10)
# look at the film_date colum, from ted website we know that the value in film_date is unix timestamp

ted.film_date.head()
#change the value in film_date to be readiable with pd.to_datetime() and put it in new colum

ted['film_datetime'] = pd.to_datetime(ted.film_date, unit='s')



#check with sample method (taking random samples) to see if it worked

ted[['event','film_datetime']].sample(5)
# check the dtype for the column film_datetime

ted.dtypes
# select the year from the column film_datetime (its cool to check dt options as well) then count the values and after that plot it

ted.film_datetime.dt.year.value_counts().plot()
# the problem with the plot above is the order of the data. the plot will graph 

#it as it shows without ordering them and to fix that we 

#need to order the output before poltting it. notice that the index in the output for ted.film_datetime.dt.year.value_counts() line is the year 



ted.film_datetime.dt.year.value_counts().sort_index().plot()

ted.event.value_counts().head()
# count the talks in each event and show the mean of views 

ted.groupby('event').views.agg(['count', 'mean']).sort_values('mean').tail()
# the results from above shows TEDxPuget Sound thas the largest mean values but also shows it has 1 talk

# only. that does not reflect that this talk is what should be the best talk

# so we add to agg() method the sum of the views and we sorted by sum 

ted.groupby('event').views.agg(['count', 'mean','sum']).sort_values('sum').tail()
# Lets have a look at rating colum

ted.ratings.head()
# lets have a look at the 1st line. you can use the code line below or you can use ted.ratings[0]

ted.loc[0,'ratings']
# what is the data type of ratings column ? is it list of dictionaries ?

type(ted.loc[0,'ratings'])
# the above shows the dtype is a str and not a list of dictioneries which we can call it a

# stringfied list of dictioeries. and here we should unpack this data. the egneric way to do this is by 

# using abstract syntax tree module and use the function called literal_eval



import ast

ast.literal_eval(ted.ratings[0])



# now we need to apply this to the entire series



ted ['ratings_list'] = ted.ratings.apply(ast.literal_eval)

ted.ratings_list[0]
# what we are trying to acomplish is to sum the count. we are going to build a function for that

# and then we use apply method



def get_num_ratings(list_of_dicts):

    num = 0

    for d in list_of_dicts:

        num = num + d['count']

    return num



# check if the function is working correctly

get_num_ratings(ted.ratings_list[0])
# use apply method

ted['num_ratings'] = ted.ratings_list.apply(get_num_ratings)
# another way

pd.DataFrame(ted.ratings_list[0])['count'].sum()
ted.ratings_list.head()
# does all talks ratings have 'Funny' rating ?

ted.ratings.str.contains('Funny').value_counts()



#Yes since we have total of 2550 rows and the output of the value_counts() is 2550
# we will write a function to return the count if funny ratings



def get_funny_ratings(list_of_dicts):

    for d in list_of_dicts:

        if d['name'] == 'Funny':

            return d['count']



# apply the function

ted['funny_ratings'] = ted.ratings_list.apply(get_funny_ratings)

ted.funny_ratings.head()
ted['funny_ratings_rate'] = ted.funny_ratings / ted.num_ratings

ted.funny_ratings_rate.head()
ted.sort_values('funny_ratings_rate').speaker_occupation.head(20)
ted.groupby('speaker_occupation').funny_ratings_rate.mean().sort_values().tail()
ted.speaker_occupation.describe()
occupation_counts = ted.speaker_occupation.value_counts()

top_occupations = occupation_counts[occupation_counts >= 5].index

ted_top_occupations = ted[ted.speaker_occupation.isin(top_occupations)]

ted_top_occupations.shape
ted_top_occupations.groupby('speaker_occupation').funny_ratings_rate.mean().sort_values()