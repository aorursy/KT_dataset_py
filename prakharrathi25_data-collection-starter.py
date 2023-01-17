# Install the praw library 

!pip install praw
# Import the library

import reddit_scraper as rs

import praw

import pandas as pd
# Credentials generated from the reddit developers applications page

# Hidden to protect my details. Add your own info.  

my_client_id = ''

my_client_secret = ''

user = ''
# Authenticate the Reddit instance

reddit = rs.reddit_auth(my_client_id, my_client_secret, user)
# NOTE:- This is currently not autheticating properly. 

# In case you get the output as none then authenticate conventionally

print(reddit)
# Conventional authentication

#reddit = praw.Reddit(client_id=my_client_id, client_secret=my_client_secret, user_agent=user)
# These are the predefined features and will be set by default 

features = [

    'ID', 

    'is_Original', 

    'Flair',

    'num_comments', 

    'Title',

    'Subreddit', 

    'Body', 

    'URL', 

    'Upvotes',

    'created_on', 

    'Comments'

]
# Set the desired subreddit 

subreddit = "depression"
# Collect data in a dataframe

data = rs.scrape_without_flairs(reddit, sub_name=subreddit, 

                                          features=features, 

                                          num_posts=100, comments=False)
data.head()
# Save Data 

data.to_csv('depression_reddit_data.csv')
# Get a list of the unique flairs associated with a subreddit.

flair_list = rs.get_unique_flairs(reddit, sub_name='India', num_posts=100)
print(flair_list)
# Scrape data with a list of flairs

data = rs.scrape_with_flairs(reddit, sub_name='India', flairs=flair_list, num_per_flair=5, features=features, comments=False)