# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib as mpl







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(os.path.join(dirname, filename))

df
df.info()
df = df[["date","tweet","hashtags","username","nlikes","nreplies","nretweets","reply_to"]]

df
# Who tweeted the most

df["username"].value_counts().plot(kind="bar",figsize=(12,7))

plt.title("Most tweets")

plt.xlabel("Presidential candidate")

plt.ylabel("Number of tweets")

plt.xticks(rotation=45)

plt.style.use(['seaborn'])
# Who got the most likes,replies and retweets

tweet_responses = df.groupby(["username"]).sum().sort_values(by=(["nlikes"]))

tweet_responses.plot(kind="bar",figsize=(12,7))

plt.title("Most interactions")

plt.xlabel("Presidential candidate")

plt.ylabel("Responses")

plt.xticks(rotation=45)

plt.style.use(['seaborn'])
# Add number of tweets to new data frame

tweet_responses["tweets"] = df["username"].value_counts()

tweet_responses
# Responses per tweets

tweet_responses["likes_per_tweet"] = round(tweet_responses["nlikes"] / tweet_responses["tweets"],0)

tweet_responses["replies_per_tweet"] = round(tweet_responses["nreplies"] / tweet_responses["tweets"],0)

tweet_responses["retweets_per_tweet"] = round(tweet_responses["nretweets"] / tweet_responses["tweets"],0)

tweet_responses
# Who got the most responses per tweet

tweet_responses_final = tweet_responses[["likes_per_tweet","replies_per_tweet","retweets_per_tweet"]]

tweet_responses_final
tweet_responses_final.plot(kind="bar",figsize=(12,7))

plt.xlabel("Presidential candidate")

plt.ylabel("Number of interactions")

plt.xticks(rotation=45)

plt.title("Most interactions per tweet")

plt.style.use(['seaborn'])
def tweet_words(word,interaction,graph=0):

    # The most tweets with the word "INSERT WORD HERE"

    tweet_df = df[["username","tweet","nlikes","nreplies","nretweets"]]

    word_df = pd.DataFrame()

    word_username_list = []

    word_tweet_list = []

    word_nlikes_list = []

    word_nreplies_list = []

    word_nretweets_list = []



    for user,tweet,nlike,nreplies,nretweets in list(zip(tweet_df["username"],tweet_df["tweet"],tweet_df["nlikes"],

                                                       tweet_df["nreplies"],tweet_df["nretweets"])):

        if word in tweet:

            word_username_list.append(user)

            word_tweet_list.append(tweet)

            word_nlikes_list.append(nlike)

            word_nreplies_list.append(nreplies)

            word_nretweets_list.append(nretweets)





    word_df["username"] = word_username_list

    word_df["tweet"] = word_tweet_list

    word_df["nlikes"] = word_nlikes_list

    word_df["nreplies"] = word_nreplies_list

    word_df["nretweets"] = word_nretweets_list

    

    # Most (interaction) tweets



    interaction_df = word_df.sort_values(by=interaction,ascending=False)    

    name = interaction_df.groupby(by="username")

    most_interaction_tweets = name[["tweet",interaction]].first()



    

    most_interaction_tweets = most_interaction_tweets.reset_index()



    for user,interact,tweet in list(zip(most_interaction_tweets["username"],

                                        most_interaction_tweets[interaction],most_interaction_tweets["tweet"])):

        print(user,"\t",interaction,interact,"\n",tweet,"\n")

        

        

    word_df = word_df[["username","nlikes","nreplies","nretweets"]]

    word_df.columns

    word_reponses = round(word_df.groupby(["username"]).sum() / word_df.groupby(["username"]).count(),0)

    print("=========================================================================================")



    if graph !=0:

        word_reponses.plot(kind="bar",figsize=(12,7))

        word = word.upper()

        plt.title("The use of the word " + word +  ": Most interactions per tweet")

        plt.xlabel("Presidential candidate")

        plt.ylabel("Interactions per tweet")

        plt.xticks(rotation=45)

        plt.style.use(['seaborn'])

        plt.show

tweet_words("youth","nlikes")

tweet_words("youth","nreplies")

tweet_words("youth","nretweets",graph=1)
tweet_words("land","nlikes")

tweet_words("land","nreplies")

tweet_words("land","nretweets",graph=1)
tweet_words("health","nlikes")

tweet_words("health","nreplies")

tweet_words("health","nretweets",graph=1)
tweet_words("corruption","nlikes")

tweet_words("corruption","nreplies")

tweet_words("corruption","nretweets",graph=1)
tweet_words("money","nlikes")

tweet_words("money","nreplies")

tweet_words("money","nretweets",graph=1)
# Tweets with most retweets

df = df.sort_values(by="nretweets",ascending=False)

name = df.groupby(by="username")

most_retweet_tweets = name[["date","tweet","nretweets"]].first()

most_retweet_tweets
most_retweet_tweets = most_retweet_tweets.reset_index()



for user,retweet,tweet in list(zip(most_retweet_tweets["username"],most_retweet_tweets["nretweets"],most_retweet_tweets["tweet"])):

    print(user,"\t Retweets:",retweet,"\n",tweet,"\n")
# Frequency of words used for all 3 candidates



from wordcloud import WordCloud, STOPWORDS 



comment_words = '' 

stopwords = set(STOPWORDS) 

  

# iterate through the csv file 

for val in df["tweet"]: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    comment_words += " ".join(tokens)+" "

  

wordcloud = WordCloud(width = 1000, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 10), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
# Frequency of hagegeingob

df_mvenaani = df.loc[df["username"] == "hagegeingob"]

  

comment_words = '' 

stopwords = set(STOPWORDS) 

  

# iterate through the csv file 

for val in df_mvenaani["tweet"]: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    comment_words += " ".join(tokens)+" "

  

wordcloud = WordCloud(width = 1000, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 10), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title("hagegeingob word frequency") 

  

plt.show() 
# Frequency of mvenaani

df_mvenaani = df.loc[df["username"] == "mvenaani"]

  

comment_words = '' 

stopwords = set(STOPWORDS) 

  

# iterate through the csv file 

for val in df_mvenaani["tweet"]: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    comment_words += " ".join(tokens)+" "

  

wordcloud = WordCloud(width = 1000, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 10), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title("mvenaani word frequency") 

  

plt.show() 
# Frequency of panduleni_itula

df_mvenaani = df.loc[df["username"] == "panduleni_itula"]

  

comment_words = '' 

stopwords = set(STOPWORDS) 

  

# iterate through the csv file 

for val in df_mvenaani["tweet"]: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    comment_words += " ".join(tokens)+" "

  

wordcloud = WordCloud(width = 1000, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 10), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title("panduleni_itula word frequency") 

  

plt.show() 