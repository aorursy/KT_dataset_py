#we will import the TED.COM dataset

import pandas as pd

import matplotlib.pyplot as plt

df=pd.read_csv('/kaggle/input/ted-talks-main-csv/ted_main.csv')

df.head()
#now we will take a look at the data

df.shape

#the output is total number of rows and columns
#this shows the datatypes

df.dtypes

#the output object includes strings, list,dicts
#now we will check the missing values in the entire dataset

df.isna().sum()

#the output tells that the speaker occupation has 6 null values, we shall handle it later
#now we will find out which talks provoke most online discussion

#one method is that the the talks with more online comments/subcomments is provocative

#next method is that we shall form a separate dataframe which calculates comments/total number of views

df.sort_values('comments').tail()

df['comments_per_view']=df.comments/df.views
df['comments_per_view']
#here you can see comments_per_view attached on to the dataframe

df.sort_values('comments_per_view').tail()
# we can also interpret that 'The case for same-sex marriage' talk by Diane J Savino has more provocative

#for every view there is 0.002 comments per views

#now we will see views per comment

df['views_per_comment']=df.views/df.comments
df.sort_values('views_per_comment').head()
#Now we will visulaize the distribution of comments

df.comments.plot()



#this is not the right way normally time series data are plotted with line plots
df.comments.plot(kind='hist')
#we will make the chart most informative

df[df.comments<1000].comments.plot(kind='hist')
#one more method to do the same above plot

df.loc[df.comments<1000,'comments'].plot(kind='hist',bins=20)

#place cursor inside parenthesis and clik shift+tab and tryout different plot types
#now we will plot number of talks that took each year

df.film_date.head()

#the output is unis time stamps
pd.to_datetime(df.film_date,unit='s')
#now we will create a new datframe out of above output

df['film_datetime']=pd.to_datetime(df.film_date,unit='s')
df.head()
#now we will compare the dates output with event column

df[['event','film_datetime']].sample(5)

#seems like the output is comparable
df.dtypes
# the following attributes are available with datetime (dayofyear,year,dayofweek)

df.film_datetime.dt.dayofyear
#now we will count the number of talks every year

df.film_datetime.dt.year.value_counts()
#try plotting by removing sort_index()

df.film_datetime.dt.year.value_counts().sort_index().plot()
#now we will try to find the best event in the TED history

#one method is that number od events in an year

df.event.value_counts()
#mean number of views for each event

df.groupby('event').views.mean().sort_values().tail()
#now we will pass on the aggregate function



df.groupby('event').views.agg(['count','mean','sum']).sort_values('sum').tail()
#now we will unpack the ratings data

df['ratings']
#what is the datatypes

type(df.ratings[0])

#the dtypeis a stringified dictionary
df.ratings[0]
import ast

#abstract syntax tree
ast.literal_eval(df.ratings[0])

#its a magic function which outputs string as a list
#now we will pass the custom function which converts the string to list

def str_to_list(ratings_str):

    return ast.literal_eval(ratings_str)

str_to_list(df.ratings[0])
#now the string output is converted to list

type(str_to_list(df.ratings[0]))
df.ratings.apply(str_to_list).head()
#the above function can also be done by using a Lamda function

#try changing ast.literal_eval to str_to_list

df['ratings_list']=df.ratings.apply(lambda x:ast.literal_eval(x))
#the goal is to convert the series of strings to series of lists

df['ratings_list']
#rating converted into ratings_list(series)

df.head(1)
#now we will try to count the total number of ratings received by each talk

df.ratings_list[0]
#now we will buind the function to count the ratings

def get_num_ratings(list_of_dicts):

    return list_of_dicts[0]['count']
get_num_ratings(df.ratings_list[0])
#now we will and the number of counts for the above function

def get_num_ratings(list_of_dicts):

    num=0

    for d in list_of_dicts:

        num=num+d['count']

        return num

        

    
#so the output is the total count of ratings of each talk

get_num_ratings(df.ratings_list[0])
#we will create a separate column in the dataframe

df['num_ratings']=df.ratings_list.apply(get_num_ratings)
df['num_ratings']
df.head(1)

#one more method to calculate the sum of counts of review

pd.DataFrame(df.ratings_list[0])['count'].sum()
#now we will find which occupations deliver the funniest ted talks

#now we will count the funny ratings

df.ratings.str.contains('funny').value_counts()
#now we will get the funny ratings from the dicts

def get_funny_ratings(list_of_dicts):

    for d in list_of_dicts:

        if d['name']== 'Funny':

            return d['count']
df['funny_ratings']=df.ratings_list.apply(get_funny_ratings)
df['funny_ratings'].head()
#now we will calculate the percentage of ratings which are funny

df['funny_rate']=df.funny_ratings/df.num_ratings
df.head(3)
df.sort_values('funny_rate').speaker_occupation

#this output gives the funny rate with respect to the occupations
#now we will analyse the funny rate by occupation

df.groupby('speaker_occupation').funny_rate.mean().sort_values()
df.speaker_occupation.describe()
#the speaker occupations which are well reperesented

occupation_counts=df.speaker_occupation.value_counts()
#now we will filter above series by their values

top_occupations=occupation_counts[occupation_counts>=5].index

#now we will filter out top occupations which are represented more than 5 times in the dataframe

ted_top_occupations=df[df.speaker_occupation.isin(top_occupations)]

top_occupations
ted_top_occupations
ted_top_occupations.shape