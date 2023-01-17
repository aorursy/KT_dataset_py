import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

import seaborn as sns
#Load and inspect data

df = pd.read_csv('../input/Tweets.csv')

df.info()
df.head()
def clean_df(df):

    df = df.loc[: , ['airline_sentiment', 

                         'airline_sentiment_confidence',

                         'negativereason',

                         'negativereason_confidence',              

                         'name',

                         'text',

                         'tweet_coord',

                         'tweet_created',

                         'airline']].rename(columns = {'airline_sentiment':'Rating',

                                                             'airline_sentiment_confidence':'Rating_Conf',

                                                             'negativereason':'Negative_Reason',

                                                             'negativereason_confidence':'Reason_Conf',

                                                             'name':'User',

                                                             'text':'Text',

                                                             'tweet_coord':'Coordinates',

                                                             'tweet_created':'Date'}).set_index('Date')

    return df

clean_df(df).head(10)
#Groupby airline, and reference the ratings column and then extract total count

print(clean_df(df).groupby('airline')['Rating'].count())

#groupby both airlines and rating and extract total count

print(clean_df(df).groupby(['airline','Rating']).count().iloc[:,0])
#create a graph by calling our clean_data function and then plots the total number of each tweet rating (positive,negative, or neutral)

ax = clean_df(df).groupby(['airline','Rating']).count().iloc[:,0].unstack(0).plot(kind = 'bar', title = 'Airline Ratings via Twitter')

ax.set_xlabel('Rating')

ax.set_ylabel('Rating Count')





plt.show()

#Count of all tweet ratings for each airline (negative, neutral, positive)

itemized_tweets = clean_df(df).groupby(['airline','Rating']).count().iloc[:,0]

#Negative tweet total index for each airline:

#American 0

#Delta 3

#southwest 6

#US Airways 9

#United 12

#Virgin 15



#Count of total tweets about an airline

total_tweets = clean_df(df).groupby(['airline'])['Rating'].count()

#Airline index in total tweets:

#American 0

#Delta 1

#Southwest 2

#US Airways 3

#United 4

#Virgin 5





#Create a dictionary of percentage of negative tweets = (negative_tweets / total_tweets)

my_dict = {'American':itemized_tweets[0] / total_tweets[0],

           'Delta':itemized_tweets[3] / total_tweets[1],

           'Southwest': itemized_tweets[6] / total_tweets[2],

           'US Airways': itemized_tweets[9] / total_tweets[3],

           'United': itemized_tweets[12] / total_tweets[4],

           'Virgin': itemized_tweets[15] / total_tweets[5]}



#make a dataframe from the dictionary

perc_negative = pd.DataFrame.from_dict(my_dict, orient = 'index')

#have to manually set column name when using .from_dict() method

perc_negative.columns = ['Percent Negative']

print(perc_negative)

ax = perc_negative.plot(kind = 'bar', rot=0, colormap = 'Blues_r', figsize = (15,6))

ax.set_xlabel('Airlines')

ax.set_ylabel('Percent Negative')

plt.show()

itemized_tweets = clean_df(df).groupby(['airline','Rating']).count().iloc[:,0]

#Positve tweet total index for each airline:

#American 2

#Delta 5

#southwest 8

#US Airways 11

#United 14

#Virgin 17



total_tweets = clean_df(df).groupby(['airline'])['Rating'].count()

#Airline index in total tweets:

#American 0

#Delta 1

#Southwest 2

#US Airways 3

#United 4

#Virgin 5





#Create a dictionary of percentage of positive tweets = (positive_tweets / total_tweets)

my_dict = {'American':itemized_tweets[2] / total_tweets[0],

           'Delta':itemized_tweets[5] / total_tweets[1],

           'Southwest': itemized_tweets[8] / total_tweets[2],

           'US Airways': itemized_tweets[11] / total_tweets[3],

           'United': itemized_tweets[14] / total_tweets[4],

           'Virgin': itemized_tweets[17] / total_tweets[5]}



#make a dataframe from the dictionary

perc_positive = pd.DataFrame.from_dict(my_dict, orient = 'index')

#have to manually set column name when using .from_dict() method

perc_positive.columns = ['Percent Positive']

print(perc_positive)

ax = perc_positive.plot(kind = 'bar', rot=0, colormap = 'Blues_r', figsize = (15,6))

ax.set_xlabel('Airlines')

ax.set_ylabel('Percent Positve')

plt.show()

#create a function that will concatenate our perc_negative, perc_neutral and perc_positive dataframes into one single dataframe

def merge_dfs(x,y,z):

    #generate a list of the dataframes

    list_of_dfs = [x,y,z]

    #concatenate the dataframes, axis = 1 because they all have the same index, we just want to add the columns together

    concatenated_dataframe = pd.concat(list_of_dfs, axis = 1)

    return concatenated_dataframe
itemized_tweets = clean_df(df).groupby(['airline','Rating']).count().iloc[:,0]

#Netural tweet total index for each airline:

#American 1

#Delta 4

#southwest 7

#US Airways 10

#United 13

#Virgin 16



total_tweets = clean_df(df).groupby(['airline'])['Rating'].count()

#Airline index in total tweets:

#American 0

#Delta 1

#Southwest 2

#US Airways 3

#United 4

#Virgin 5





#Create a dictionary of percentage of positive tweets = (positive_tweets / total_tweets)

my_dict = {'American':itemized_tweets[1] / total_tweets[0],

           'Delta':itemized_tweets[4] / total_tweets[1],

           'Southwest': itemized_tweets[7] / total_tweets[2],

           'US Airways': itemized_tweets[10] / total_tweets[3],

           'United': itemized_tweets[13] / total_tweets[4],

           'Virgin': itemized_tweets[16] / total_tweets[5]}



#make a dataframe from the dictionary

perc_neutral = pd.DataFrame.from_dict(my_dict, orient = 'index')

#Have to manually set column name

perc_neutral.columns = ['Percent Neutral']



#call our function to concatenate all 3 dataframes of percentages

percentage = merge_dfs(perc_neutral, perc_negative, perc_positive)

print(percentage)



#graph all of our data

ax = percentage.plot(kind = 'bar', stacked = True, rot = 0, figsize = (15,6))

#set x label

ax.set_xlabel('Airlines')

#set y label

ax.set_ylabel('Percentages')

#move the legend to the bottom of the graph since it wants to sit over all of our data and block it - stupid legend

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),

          fancybox=True, shadow=True, ncol=5)



plt.show()
observation = list(clean_df(df).reset_index().iloc[6750:6755,8])

tweet_text = list(clean_df(df).reset_index().iloc[6750:6755,6])



for pos, item in enumerate(observation):

    print('Airline as compiled: ' + str(item))

    print('The actual tweet text: ')

    print(tweet_text[pos], '\n''\n')
#let's start by getting rid of the current 'Airline' column

#'Airline' was the last column (8) so we just sliced the dataframe, stopping at column (7) 'Coordinates'

new_df = clean_df(df).iloc[:,0:7]

new_df.head()
#first, create a new column called 'Airline'

#Then reference the 'text' column to apply your regular expression function to

#apply a lambda function that parses through each tweet text and searches for '@' symbol followed by any letter type

#extract the first matched instance [0] in the event there are multiple

new_df['Airline'] = new_df.Text.apply(lambda x: re.findall('\@[A-Za-z]+', x)[0])



#check that our regular expression is working

list(new_df.Airline.head(10))
#get all unique twitter tags and the count for how many times it appears in the column

twitter_tags = np.unique(new_df.Airline, return_counts = True)



#compile twitter_tags so that it lists the unique tag and its total count side by side instead of 2 seperate arrays

twitter_tags_count = list(zip(twitter_tags[0],twitter_tags[1]))

twitter_tags_count
#List of all airlines in the data as found from the tweets in search above

airline_list = ['@virginamerica','@united','@southwestair','@americanair','@jetblue','@usairways']

    

#compile a regex search to seperate out only the airline tag and ignoring other users tags in the text

#using the compile method is an easier way to input our "match" pattern into the search engine, especially in this event

#when we are searching for mulitple airlines.

#we are ignoring case, or capitaliztion  in order to negate all the uniquess we encountered in the list above

airlines = re.compile('|'.join(airline_list), re.IGNORECASE)

    

#apply the compiled regex search and remove the twitter tag '@'

#for example, the following code takes @AmericanAir and returns AmericanAir

new_df['Airline'] = new_df.Airline.apply(lambda x: np.squeeze(re.findall(airlines, x))).str.split('@').str[1]

print(list(new_df.Airline.head(10)))
no_airline = new_df.reset_index()

no_airline = no_airline[no_airline.Airline.isnull()].Text.apply(lambda x: re.findall('\@[A-Za-z]+', x))

no_airline
#reset the index of our dataframe

new_df = new_df.reset_index()



#compile a list of index locations of the tweets that return null and set their airline value to the appropriate

#airline referenced in the tweet

united = [737,868,1088,4013]

southwest = [4604,5614,5615,6136,6362]

jetblue = [6796,6811,6906]

usairways = [7330, 8215,10243,10517,10799,10864,10874,10876,11430]

american = [11159,12222,12417,12585,13491,13979]

delta = [12038, 12039]

new_df.set_value(united,'Airline','united')

new_df.set_value(southwest,'Airline','southwestair')

new_df.set_value(jetblue,'Airline','jetblue')

new_df.set_value(usairways,'Airline','usairways')

new_df.set_value(american,'Airline','americanair')

new_df.set_value(delta,'Airline','delta')

    

#Since all airlines tweets are camel case in different orders, make all airlines uppercase so they are all equal

new_df.Airline = new_df.Airline.apply(lambda x: x.upper())

    

#create a dictionary to map the all uppercase airlines to the proper naming convention

map_airline = {'AMERICANAIR':'American Airlines',

                'JETBLUE':'Jet Blue',

                'SOUTHWESTAIR':'Southwest Airlines',

                'UNITED': 'United Airlines',

                'USAIRWAYS': 'US Airways',

                'VIRGINAMERICA':'Virgin Airlines',

                'DELTA':'Delta Airlines'}

    

#map the uppercase airlines to the proper naming convention

new_df.Airline = new_df.Airline.map(map_airline)



#display our new airlines!!!

np.unique(new_df.Airline)
rating = list(new_df.Rating)

conf = list(new_df.Rating_Conf)

text = list(new_df.Text)



for i in range(10):

    print(rating[i], '\n', conf[i], '\n', text[i],'\n','\n')

    

    
#we could make this one line, but i'm breaking it up for readability

#set our boolean variable so that it filters the dataframe for only instances where the rating conf is >0.51





conf_df = new_df[new_df.Rating_Conf >= 0.51 ]

print(conf_df.info())

conf_df.head(10)
#create a copy of our original dataframe and reset the index

date = conf_df.reset_index()

#convert the Date column to pandas datetime

date.Date = pd.to_datetime(date.Date)

#Reduce the dates in the date column to only the date and no time stamp using the 'dt.date' method

date.Date = date.Date.dt.date

date.Date.head()

conf_df = date

conf_df.head()
test = conf_df[conf_df.Airline != 'Delta Airlines'].groupby(['Airline','Rating']).count().iloc[:,0]

test
def percentages(df, rating = 'negative'):

    if rating == 'negative':

        i = 0

        column = 'Percent Negative'

    elif rating == 'neutral':

        i = 1

        column = 'Percent Neutral'

    elif rating == 'positive':

        i = 2

        column = 'Percent Positive'

        

    #Count of all tweet ratings for each airline (negative, neutral, positive), remove Delta since it only has 2 entries total

    itemized_tweets = df[df.Airline != 'Delta Airlines'].groupby(['Airline','Rating']).count().iloc[:,0]

    #Rating tweet total index for each airline:

    #American i

    #Jet Blue i + 3

    #southwest i + 6

    #US Airways i + 9

    #United i + 12

    #Virgin i + 15



    #Count of total tweets about an airline

    total_tweets = df[df.Airline != 'Delta Airlines'].groupby(['Airline'])['Rating'].count()

    #Airline index in total tweets:

    #American 0

    #Jet Blue 1

    #Southwest 2

    #US Airways 3

    #United 4

    #Virgin 5





    #Create a dictionary of percentage of rating tweets = (rating_tweets / total_tweets)

    my_dict = {'American':itemized_tweets[i] / total_tweets[0],

                'Jet Blue':itemized_tweets[i + 3] / total_tweets[1],

                'Southwest': itemized_tweets[i + 6] / total_tweets[2],

                'US Airways': itemized_tweets[i + 9] / total_tweets[3],

                'United': itemized_tweets[i + 12] / total_tweets[4],

                'Virgin': itemized_tweets[i + 15] / total_tweets[5]}



    #make a dataframe from the dictionary

    perc_df = pd.DataFrame.from_dict(my_dict, orient = 'index')

        

    #have to manually set column name when using .from_dict() method

    perc_df.columns = [column]

        

    return perc_df

    

#Create a df called negative that contains the percent negatives by calling the function above

negative = percentages(conf_df, 'negative')



#Create a df called neutral that contains the percent neutrals by calling the function above

neutral = percentages(conf_df, 'neutral')



#Create a df called positive that contains the percent positives by calling the function above

positive = percentages(conf_df, 'positive')



#call the earlier function that merges all 3 data frames into one

merged_perc = merge_dfs(negative, positive, neutral)

merged_perc
ax = merged_perc.plot(kind = 'bar', rot = 0, figsize = (15,6))

plt.show()
#function that reduces the dataframe to only the airline and the negative reasons, then extract the reasons and the frequency

#each reason was referenced to an airline

def reason(df):

    df = df.reset_index().loc[:,['Airline','Negative_Reason']].dropna().groupby(['Airline','Negative_Reason']).size()

    return df



#call the function and plot the results

ax1 = reason(conf_df).unstack(0).plot(kind = 'bar', figsize = (15,6), rot = 70)



plt.show()
print(conf_df.Date.min())

print(conf_df.Date.max())
#groupby by Date first making it the main index, then group by the airline, then finally the rating and see how many

#of each rating an airline got for each date

day_df = conf_df.groupby(['Date','Airline','Rating']).size()

day_df
day_df = day_df.reset_index()

day_df.head()
#rename the column

day_df = day_df.rename(columns = {0:'Count'})

#filter to only negative ratings

day_df = day_df[day_df.Rating == 'negative'].reset_index()

#Remove delta since it only has 2 entries

day_df = day_df[day_df.Airline != 'Delta Airlines']

day_df.head()
#slice out the first 2 columns of the resultant dataframe

day_df = day_df.iloc[:,1:5]



#groupby and plot data

ax2 = day_df.groupby(['Date','Airline']).sum().unstack().plot(kind = 'bar', colormap = 'viridis', figsize = (15,6), rot = 70)

labels = ['American Airlines','Jet Blue','Southwest Airlines','US Airways','United Airlines','Virgin Airlines']

ax2.legend(labels = labels)

ax2.set_xlabel('Date')

ax2.set_ylabel('Negative Tweets')

plt.show()