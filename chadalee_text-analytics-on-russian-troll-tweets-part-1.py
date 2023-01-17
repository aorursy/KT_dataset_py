# Basic library loading
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
%pylab inline
# Read in the dataset required
troll = pd.read_csv('../input/tweets.csv')
print(troll.shape)
troll.head(2)
troll.isnull().sum().sort_values(ascending = False)
# drop NAs in the text column and update the troll dataframe
troll.dropna(subset = ['text'], inplace = True)
print(troll.dtypes)
# convert created_str to datetime format
troll['created_str'] = pd.to_datetime(troll['created_str'])

# convert ids to object datatype
columns = ['user_id', 'tweet_id', 'retweeted_status_id', 
           'retweeted_status_id', 'in_reply_to_status_id']

for column in columns:
    troll[column] = troll[column].astype('object')
troll.dtypes
start_date_tweet = troll['created_str'].min()
end_date_tweet = troll['created_str'].max()

print(start_date_tweet, end_date_tweet)
# created_str_data holds the date component of the created_str column
troll['created_str_date'] = pd.to_datetime(troll['created_str'].dt.date)
# Count the number of times a date appears in the dataset and convert to dataframe
tweet_trend = pd.DataFrame(troll['created_str_date'].value_counts())

# index is date, columns indicate tweet count on that day
tweet_trend.columns = ['tweet_count']

# sort the dataframe by the dates to have them in order
tweet_trend.sort_index(ascending = True, inplace = True)
# make a line plot of the tweet count data and give some pretty labels! ;)
# the 'rot' argument control x-axis ticks rotation
plt.style.use('seaborn-darkgrid')
tweet_trend['tweet_count'].plot(linestyle = "-", figsize = (12,8), rot = 45, color = 'k',
                               linewidth = 1)
plt.title('Tweet counts by date', fontsize = 15)
plt.xlabel('Date', fontsize = 13)
plt.ylabel('Tweet Count', fontsize = 13)
# these are dates corresponding to important dates from the trump campaign.
dates_list = ['2015-06-16', '2015-12-07', '2016-02-01',
              '2016-03-01', '2016-03-03', '2016-03-11',
              '2016-05-03', '2016-05-26', '2016-06-20', 
              '2016-07-15', '2016-07-21', '2016-08-17',
              '2016-09-01', '2016-10-07', '2016-11-08']

# create a series of these dates.
important_dates = pd.Series(pd.to_datetime(dates_list))

# add columns to identify important events, and mark a 0 or 1.
tweet_trend['Important Events'] = False
tweet_trend.loc[important_dates, 'Important Events'] = True
tweet_trend['values'] = 0
tweet_trend.loc[important_dates, 'values'] = 1
# plot the line chart for trend, a monthly average of tweet counts and add red dots to 
# mark important events.
plt.style.use('seaborn-darkgrid')
tweet_trend['tweet_count'].plot(linestyle = "--", 
                                figsize = (12,8), rot = 45, 
                                color = 'k',
                                label = 'Tweet Count per Day',
                               linewidth = 1)

# plot dots for where values in the tweet_trend df are 1
plt.plot(tweet_trend[tweet_trend['Important Events'] == True].index.values,
         tweet_trend.loc[tweet_trend['Important Events'] == True, 'values'],
         marker = 'o', 
         color = 'r',
         linestyle = 'none',
        label = 'Important Dates in campaign')

# Lets add a 30 day moving average on top to view the trend! Min_periods tells rolling() to
# use 10 points if 30 not available!
plt.plot(tweet_trend['tweet_count'].rolling(window = 30, min_periods = 10).mean(), 
         color = 'r', 
         label = '30 Day Moving Avg # of tweets')
plt.title('Tweet counts by date', fontsize = 15)
plt.xlabel('Date', fontsize = 13)
plt.ylabel('Tweet Count', fontsize = 13)
plt.legend(loc = 'best')
# Calculate the percentage change in tweet counts
tweet_trend['Pct_Chg_tweets'] = tweet_trend['tweet_count'].pct_change()*100

# Lets see values only for the important dates. This Pct_Chg_tweets shows us the percentage
# change in tweets for the day of the event versus the previous day!
tweet_trend.loc[tweet_trend['values'] == 1,['tweet_count', 'Pct_Chg_tweets']]
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# line plot of the percentage change in tweet counts
tweet_trend['Pct_Chg_tweets'].plot(linestyle = "--", figsize = (12,8), rot = 45, 
                                   color = 'k',
                                  linewidth = 1)
# add the dots for important events!
plt.plot(tweet_trend[tweet_trend['Important Events'] == True].index.values,
         tweet_trend.loc[tweet_trend['Important Events'] == True, 'values'],
         marker = 'o', 
         color = 'r',
         linestyle = 'none')
plt.title('Tweet count change', fontsize = 15)
plt.xlabel('Date', fontsize = 13)
plt.ylabel('Tweet Count Change', fontsize = 13)
# take a look at what the 'text' column holds
troll['text'].head(10)
# define a function that takes in a tweet and throws out the text without the RT.
def remove_retweet(tweet):
    '''Given a tweet, remove the retweet element from it'''
    text_only = []
    if len(re.findall("^RT.*?:(.*)", tweet)) > 0:
        text_only.append(re.findall("^RT.*?:(.*)", tweet)[0])
    else:
        text_only.append(tweet)
    return text_only[0]

# extract texts and place in a list
text_only = troll['text'].map(remove_retweet)
# this method checks for links and removes these from the tweet provided!
def remove_links(tweet):
    '''Provide a tweet and remove the links from it'''
    text_only = []
    if len(re.findall("(https://[^\s]+)", tweet)) > 0:
        tweet = re.sub("(https://[^\s]+)", "", tweet)
    if len(re.findall("(http://[^\s]+)", tweet)) > 0:
        tweet = re.sub("(http://[^\s]+)", "", tweet)    
    text_only.append(tweet)
    return text_only[0]

text_no_links = text_only.map(remove_links)
def remove_hashtags(tweet):
    '''Provide a tweet and remove hashtags from it'''
    hashtags_only = []
    if len(re.findall("(#[^#\s]+)", tweet)) > 0:
        tweet = re.sub("(#[^#\s]+)", "", tweet) 
    hashtags_only.append(tweet)
    return hashtags_only[0]

text_all_removed = text_no_links.map(remove_hashtags)
def remove_extraneous(tweet):
    '''Given a text, remove unnecessary characters from the beginning and the end'''
    tweet = tweet.rstrip()
    tweet = tweet.lstrip()
    tweet = tweet.rstrip(")")
    tweet = tweet.lstrip("(")
    tweet = re.sub("\.", "", tweet)
    return tweet

text_clean = text_all_removed.map(remove_extraneous)
# in case no mention present, we return "0"
def extract_mentions(tweet):
    '''Given a tweet, this function returns the user mentions'''
    mentions = []
    if len(re.findall('@[^\s@]+', tweet))>0:
        mentions.append(re.findall('@([^\s@]+)', tweet))
    else:
        mentions.append(["0"])
    return mentions[0]

# Put the user mentions in a new column in our dataframe
troll['user_mentions'] = text_clean.map(extract_mentions)
# Now lets remove the mentions from the tweet text
def remove_mentions(tweet):
    '''Given a text, remove the user mentions'''
    mentions = []
    if len(re.findall('@[^\s@]+', tweet))>0:
        tweet = re.sub('@[^\s@]+', "" , tweet)
        mentions.append(tweet)
    else:
        mentions.append(tweet)
    return mentions[0]

text_clean_final = text_clean.map(remove_mentions)
troll['tweet_text_only'] = text_clean_final
# in case hashtags are not found, we will use "0" as the placeholder
def extract_hashtags(tweet):
    '''Provide a tweet and extract hashtags from it'''
    hashtags_only = []
    if len(re.findall("(#[^#\s]+)", tweet)) > 0:
        hashtags_only.append(re.findall("(#[^#\s]+)", tweet))
    else:
        hashtags_only.append(["0"])
    return hashtags_only[0]

# make a new column to store the extracted hashtags and view them!
troll['tweet_hashtags'] = troll['text'].map(extract_hashtags)
troll['tweet_hashtags'].head(10)
# create a list of all hashtags
all_hashtags = troll['tweet_hashtags'].tolist()

# Next we observe that our all_hashtags is a list of lists...lets change that
cleaned_hashtags = []
for i in all_hashtags:
    for j in i:
            cleaned_hashtags.append(j)

# Convert cleaned_hashtags to a series and count the most frequent occuring
cleaned_hashtag_series = pd.Series(cleaned_hashtags)
hashtag_counts = cleaned_hashtag_series.value_counts()
# Get hashtag terms from the series and convert to list
hashes = cleaned_hashtag_series.values
hashes = hashes.tolist()

# convert list to one string with all the words
hashes_words = " ".join(hashes)

# generate the wordcloud. the max_words argument controls the number of words on the cloud
from wordcloud import WordCloud
wordcloud = WordCloud(width= 1600, height = 800, 
                      relative_scaling = 1.0, 
                      colormap = "Blues",
                     max_words = 100).generate(hashes_words)

plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
plt.style.use('seaborn-darkgrid')
plt.figure(figsize = (12,8))
plt.barh(y = hashtag_counts[1:21].index.values, width = hashtag_counts[1:21])
plt.title("Top 20 Hashtags used in Troll tweets", fontsize = 15)
plt.xlabel('Count of hashtags', fontsize = 13)
plt.ylabel('Hashtags', fontsize = 13)
# Create a dataframe with just the date and the hashtags in the tweet on that date
hashtag_date_df = troll[['created_str_date', 'tweet_hashtags']]
hashtag_date_df = hashtag_date_df.reset_index(drop = True)

# extract a list of hashtags from the dataframe
all_hashtags = hashtag_date_df['tweet_hashtags'].tolist()

hashtag_date_df.head()
# For the top 6 hashtags, lets calculate how many times that appears against each date!
count_dict = {}
for i in hashtag_counts.index.values[1:7]:
    count_hash = []
    for j in all_hashtags:
        count_hash.append(j.count(i))
    count_dict[i] = count_hash
# create a dataframe from the hashtags
hashtag_count_df = pd.DataFrame(count_dict)

# concatenate this dataframe with the hashtag_count_df
hashtag_count_df = pd.concat([hashtag_date_df, hashtag_count_df], axis = 1)
hashtag_count_df.head()
# change the created_str column into datetime format and extract just the date from it
hashtag_count_df['created_str_date'] = pd.to_datetime(hashtag_count_df['created_str_date'])

# set the index so as to plot the time series
hashtag_count_df.set_index('created_str_date', inplace = True)

# get a monthly sum of the tweets for each of these hashtags
hashtag_count_df_pivot = hashtag_count_df.resample('M').sum()

# replace 0 with nan so that these can be removed in rows where they are all NaNs
hashtag_count_df_pivot.replace(0, np.nan, inplace = True)
hashtag_count_df_pivot.dropna(how = 'all', inplace = True, axis = 0)

# replace NaNs back by 0s so that we can plot
hashtag_count_df_pivot.replace(np.nan, 0, inplace = True)
hashtag_count_df_pivot
plt.style.use('seaborn-darkgrid')
# create a 3 by 2 subplot to hold the trend of all hashtags
figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = subplots(nrows = 3,
                                                       ncols = 2,
                                                       sharey = True,
                                                       figsize = (10,8))

plt.subplots_adjust(top = 1, hspace = 0.9)
hashtag_count_df_pivot['#politics'].plot(linestyle = "-", marker = "o", color = "green",ax = ax1)
ax1.set_title("#POLITICS", fontsize = 10)
ax1.set_xlabel('Date', fontsize = 12)

hashtag_count_df_pivot['#tcot'].plot(linestyle = "-", marker = "o", color = "red", ax = ax2)
ax2.set_title("#TCOT", fontsize = 10)
ax2.set_xlabel('Date', fontsize = 12)

hashtag_count_df_pivot['#MAGA'].plot(linestyle = "-", marker = "o", color = "orange", ax = ax3)
ax3.set_title("#MAGA", fontsize = 10)
ax3.set_xlabel('Date', fontsize = 12)

hashtag_count_df_pivot['#PJNET'].plot(linestyle = "-", marker = "o", color = "blue",ax = ax4)
ax4.set_title("#PJNET", fontsize = 10)
ax4.set_xlabel('Date', fontsize = 12)

hashtag_count_df_pivot['#news'].plot(linestyle = "-", marker = "o", color = "grey", ax = ax5)
ax5.set_title("#NEWS", fontsize = 10)
ax5.set_xlabel('Date', fontsize = 12)

hashtag_count_df_pivot['#Trump'].plot(linestyle = "-", marker = "o", color = "maroon", ax = ax6)
ax6.set_title("#TRUMP", fontsize = 10)
ax6.set_xlabel('Date', fontsize = 12)
troll['user_mentions'].head(10)
user_mention = troll.loc[:, ['user_key', 'user_mentions']]
user_mention.head(6)
row_remove_mask = user_mention['user_mentions'].map(lambda x: "0" in x)
np.sum(row_remove_mask)
# keep rows where row_remove_mask is FALSE
user_mention_df = user_mention.loc[~row_remove_mask, :]
user_mention_df.reset_index(drop = True, inplace = True)
user_mention_df.head(10)
# for each row, create a one-to-one tuple of user and his user mention
new_list = []
for i in range(len(user_mention_df)):
    for j in user_mention_df.loc[i, "user_mentions"]:
        (a,b) = (user_mention_df.loc[i, 'user_key'], j)
        new_list.append((a,b))
user_mention_clean_df = pd.DataFrame({"User_Key": [a for (a,b) in new_list],
                                     "User_Mention": [b for (a,b) in new_list]})
user_mention_clean_df.head()
# create a df with user and hashtags in one tweet
user_hashtag_df = troll[['user_key', 'tweet_hashtags']]
user_hashtag_df = user_hashtag_df.reset_index(drop = True)
# Lets remove the rows where no hashtags were used
row_remove_mask = user_hashtag_df['tweet_hashtags'].map(lambda x: "0" in x)

# Remove these rows from the user hashtag df
user_hashtag_df_clean = user_hashtag_df.loc[~row_remove_mask, :]
user_hashtag_df_clean.reset_index(drop = True, inplace = True)
user_hashtag_df_clean.head()
# separate out all hashtags used.
all_hashtags = user_hashtag_df_clean['tweet_hashtags']
# count_dict = {}
# count_df = pd.DataFrame()
# for i in range(len(hashtag_counts.index.values)):
#     count_hash = all_hashtags.map(lambda x: x.count(hashtag_counts.index.values[i]))
#     count_dict[i] = count_hash
#     if i == 5000:
#         count_df = pd.DataFrame(count_dict)
#         count_dict = {}
#     elif i % 5000 == 0:
#         count_df = pd.concat([count_df, pd.DataFrame(count_dict)])
#         count_dict = {}
#     else:
#         next
# get hashtags that qualify - present in 50 or more tweets
qualify_hashtags_mask = (hashtag_counts >= 50)
qualify_hashtags = hashtag_counts[qualify_hashtags_mask]

# remove the "0" hashtags
qualify_hashtags = qualify_hashtags.drop(labels = "0")
qualify_hashtags.head()
# lets count the number of times these qualified hashtags appear in the tweets with hashtags
count_dict = {}

for i in qualify_hashtags.index.values:
    count_hash = all_hashtags.map(lambda x: x.count(i))
    count_dict[i] = count_hash

# create a dataframe from the hashtags and their counts in tweets
hashtag_count_df = pd.DataFrame(count_dict)

# concatenate this dataframe with the hashtag_count_df
user_hashtag_count_df = pd.concat([user_hashtag_df_clean, hashtag_count_df], axis = 1)
# group by user_key and get the sum of times they have used a hashtag
user_hashtag_group = user_hashtag_count_df.groupby('user_key').agg('sum').reset_index()
user_hashtag_group.head()
user_tweet_df = troll.loc[:, ['user_key', 'tweet_text_only']]
user_tweet_df.head()
users = pd.read_csv('../input/users.csv')
users.head(2)
# First we get a count of users from each time-zone and language combination!
user_loc_lang = users.groupby(['time_zone', 'lang'])['id'].agg('count').reset_index()
user_loc_lang.rename(columns = {'id':'user_count'}, inplace = True)
user_loc_lang.head(5)
# This is a custom package installed within kaggle kernel
from pySankey import sankey
sankey.sankey(user_loc_lang['time_zone'],
              user_loc_lang['lang'],
              leftWeight = user_loc_lang['user_count'],
              rightWeight = user_loc_lang['user_count'], 
              fontsize = 10)
plt.title("User profile")
# First we convert the created_at to datetime and then extract the date from this
users['created_at'] = pd.to_datetime(users['created_at'])
users['created_at_date'] = pd.to_datetime(users['created_at'].dt.date)

users['created_at_date'].head()
user_created = users.groupby('created_at_date')['id'].agg('count')

plt.style.use('fivethirtyeight')
user_created.resample('W',kind = 'period').sum().\
plot(linestyle = '-', figsize = (10,8), linewidth = 1)
title('Troll User Account Created')
xlabel('Dates')
ylabel('Count of accounts created')
user_tweet_count = troll.groupby('user_id')['text'].agg('count').reset_index()
user_tweet_count.rename(columns = {'text':'Tweet_count'}, inplace = True)
user_tweet_count_df = user_tweet_count.merge(users,
                                      left_on = 'user_id',
                                      right_on = 'id')
user_tweet_count_df.head(2)
plt.style.use('seaborn-darkgrid')
user_tweet_count_df[['name', 'Tweet_count']].sort_values('Tweet_count', ascending = False)[:10].\
set_index('name').plot(kind = 'barh', figsize = (10,8))
title('User Wise Tweet Count', fontsize = 15)
xlabel('Tweet Count', fontsize = 13)
ylabel('User Name', fontsize = 13)
correl = user_tweet_count_df['Tweet_count'].corr(user_tweet_count_df['followers_count'])
print("{0:.2f}".format(correl))
# Drawing a scatterplot of the tweet count with number of followers
fig = plt.figure(figsize = (10,8))
plt.style.use('seaborn-darkgrid')
plt.scatter(user_tweet_count_df['Tweet_count'], 
        user_tweet_count_df['followers_count'],
       marker = 'o',
       alpha = 0.5)
plt.title("Followers vs Number of Tweets", fontsize = 15)
plt.xlabel("Number of Tweets", fontsize = 13)
plt.ylabel("Follower Count", fontsize = 13)
plt.text(6000, 80000, s = "Correlation is: {0:.2f}".format(correl), fontsize = 15)
user_tweet_count_df['lang'].value_counts()
user_tweet_count_df[['name', 'lang', 'followers_count']].sort_values('followers_count', 
                                                               ascending = False)[:10]
# Lets write out these files as datasets so that they can be used in my next Kernel!
user_mention_clean_df.to_csv('User_Mentions.csv')
user_hashtag_group.to_csv('User_Hashtags.csv')
user_tweet_df.to_csv('User_Tweets.csv')