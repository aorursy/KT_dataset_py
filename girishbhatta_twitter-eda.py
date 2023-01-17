# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #base plotting library for plotly

from matplotlib.pyplot import figure # to set the figure size

import plotly.express as px #plotly library to produce plots

from wordcloud import WordCloud, ImageColorGenerator #wordcloud library

from nltk.tokenize import word_tokenize #word tokenizer

from nltk.probability import FreqDist # Frequency Distributor

from nltk.corpus import stopwords #stop words for data cleaning



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv")
print("The dataset has {} rows and {} columns".format(data.shape[0],data.shape[1]))

data.head()
print("------------------------------------------------------------------------------------------------------------")

print("The Datatype of each column in the dataset.\n\n")

print(data.dtypes)

print("------------------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------------------")

print("The Descriptive Statistics of the Dataset.\n\n")

print(data.describe())

print("------------------------------------------------------------------------------------------------------------")
print("Total number of Unique locations: ",data["user_location"].nunique())
print("------------------------------------------------------------------------------------------------------------")

print("Number of tweets for each of the unique location in the dataset.\n\n")

print(data["user_location"].value_counts())

print("------------------------------------------------------------------------------------------------------------")
print("The number of tweets where the user location is unkown: ", data["user_location"].isna().sum())
user_location_df = data["user_location"].value_counts().rename_axis("place").reset_index(name="counts")

user_location_df.head()
user_location_threshold_data = user_location_df[user_location_df["counts"]>25].head(50)



fig = px.bar(user_location_threshold_data,x="place",y="counts", title="Top 50 Locations tweets originate from")

fig.show()
def inspect_column(data,column):

    print("------------------------------------------------------------------------------------------------------------")

    print("Basic Preliminary Information of column '{}'\n\n".format(column))

    print("Total number of Unique ",column,"values: ",data[column].nunique())

    print("----Quick overview of the distribution of the variable------")

    print(data[column].value_counts())

    print("The number of tweets where the ", column ,"specific data is unkown : ", data[column].isna().sum())

    sub_data_df = data[column].value_counts().rename_axis(column).reset_index(name="counts")

    sub_data_threshold_df = sub_data_df[sub_data_df["counts"]>25].head(100)

    fig = px.bar(sub_data_threshold_df,x=column,y="counts", title="Distribution of values of column '{}'".format(column))

    fig.show()
inspect_column(data,"user_verified")
inspect_column(data,"hashtags")
# A small method that cleans up the hashtags and collates all the hashtags into a list.

all_hashtag_list = []



# Itertuples is much faster than iterrows

for each_row in data.itertuples():

    if not str(each_row.hashtags).lower() == "nan":

        each_hashtag = str(each_row.hashtags)

        each_hashtag = each_hashtag.strip('[]').replace("'","")

        all_hashtag_list += each_hashtag.split(",")

        

print("Total number of hashtags",len(all_hashtag_list))
hashtag_df = pd.DataFrame(all_hashtag_list,columns=["hashtags"])

hashtag_df.head()
count_df = hashtag_df["hashtags"].value_counts().rename_axis("hashtags").reset_index(name="counts")

count_df.head()
hashtag_final_count_dic = {}

for each_row in count_df.itertuples():

    if str(each_row.hashtags).strip().lower() == "covid19":

        if "covid19" not in hashtag_final_count_dic:

            hashtag_final_count_dic["covid19"] = each_row.counts

        else:

            hashtag_final_count_dic["covid19"] += each_row.counts

    else:

        hashtag_final_count_dic[str(each_row.hashtags).strip()] = each_row.counts

        

print("The aggregated hashtags count has {} hashtags".format(len(hashtag_final_count_dic)))
final_hashtag_count_df = pd.DataFrame(hashtag_final_count_dic.items(),columns=['hashtag','count'])

final_hashtag_count_df
final_df = final_hashtag_count_df.sort_values(by='count',ascending=False).head(10)

fig = px.bar(final_df,"hashtag","count",title="Top 10 used hashtags during the pandemic")

fig.show()
fig = px.box(data,y="user_followers", title="The overall distribution of user followers")

fig.show()
fig = px.box(data[(data["user_followers"]>0) & (data["user_followers"]<=15000)],y="user_followers",title="The distribution of User followers within 15000 user followers")

fig.show()
fig = px.histogram(data[(data["user_followers"]>0) & (data["user_followers"]<=15000)],x="user_followers", nbins=10, color_discrete_sequence=["red"],

                   title="Distribution of user followers with user followers less than 15000")

fig.show()
fig = px.box(data,y="user_friends")

fig.show()
fig = px.box(data[(data["user_friends"]>0) & (data["user_friends"]<=5000)],y="user_friends", title="Distribution of user friends with user friends less than 5k")

fig.show()
fig = px.histogram(data[(data["user_friends"]>0) & (data["user_friends"]<=5000)],x="user_friends", nbins=10, color_discrete_sequence=["red"],

                  title="Distribution of user friends less than 5K")

fig.show()
fig = px.box(data,y="user_favourites",title="Distribution of user favourites")

fig.show()
fig = px.box(data[(data["user_favourites"]>0) & (data["user_favourites"]<=25000)],y="user_favourites",title="Distribution of user favourites with a maximum of 25K")

fig.show()
fig = px.histogram(data[(data["user_favourites"]>0) & (data["user_favourites"]<=25000)],x="user_favourites", nbins=10, color_discrete_sequence=["red"],

                  title="Distribution of user favorites with a maximum of 25k user favorites")

fig.show()
count_df = data["source"].value_counts().rename_axis("source").reset_index(name="counts")

count_df
fig = px.bar(count_df.head(10), x='source', y='counts',title="Top 10 Sources to make tweets")

fig.show()
count_df = data["is_retweet"].value_counts()

count_df
print("There are {} unique values in the column 'is_retweet'".format(data["is_retweet"].nunique()))
# converting to Date type

data["date"] = pd.to_datetime(data["date"])
print("Date column is of '{}' type".format(data["date"].dtype))

data["date"].head()
data["day_of_tweet"] = pd.to_datetime(data['date']).dt.date

data["day_of_tweet"].head()
date_time_series = data.groupby("day_of_tweet").size().rename_axis("day_of_tweet").reset_index(name="number_of_tweets")

date_time_series
fig = px.line(date_time_series, x='day_of_tweet', y="number_of_tweets", title="Time series for number of tweets made per day")

fig.show()
# Plotting a really simple wordcloud of only the first tweet

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(data.text[0])

plt.figure(figsize=(30,30))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Lets gather all the tweet data into a variable.

final_text = "".join(each_text for each_text in data.text)

print ("There are {} words overall in tweets".format(len(final_text)))
#lets first tokenize the entire corpus

tokenized_words = word_tokenize(final_text)

print("The size of the tokenized words in the corpus is of size {}".format(len(tokenized_words)))
fdist = FreqDist(tokenized_words)

print(fdist)
# lets see the top 50 most frequently used words

fdist.most_common(50)
# plotting the top 50 words in the frequency distribution

plt.figure(figsize=(20,20))

fdist.plot(50)

plt.show()
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

tokenized_words = tokenizer.tokenize(final_text)
stop_words = set(stopwords.words("english"))

print(stop_words)
#removing stop words from our tweets

filtered_tokens = []

for each in tokenized_words:

    if (each not in stop_words) and (len(each) > 3):

        filtered_tokens.append(each)

print("Tokenised words now has {} words after removing stop words".format(len(filtered_tokens)))
fdist = FreqDist(filtered_tokens)

print(fdist)
plt.figure(figsize=(30,30))

fdist.plot(50)

plt.show()
#Combining the top 200 frequently used words in the dataset

analyse_str = " ".join([each[0] for each in fdist.most_common(200)])

print(analyse_str)
# WordCloud for the most frequently used words across the dataset

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(analyse_str)

plt.figure(figsize=(20,20))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# grouping by user_location and user_verified and gathering top 50 entries

user_loc_df = data.groupby(["user_location","user_verified"])["user_verified"].count().reset_index(name="count").sort_values(by=['count'], ascending=False).head(50)

user_loc_df.head()
# bar plot to show the user_location and count by user_verified

fig = px.bar(user_loc_df, x='user_location',y='count',color='user_verified',barmode="group",

            title="Relationship between the user locations v/s user verified")

fig.show()
fig = px.pie(user_loc_df, values='count', names='user_verified', title='Ratio of Verified acounts v/s Unverified accounts')

fig.show()
# extracting the user location and their respective hashtags

user_loc_hastag_data = data[["user_location","hashtags"]]

user_loc_hastag_data.head()
# converting the dataframe to dictionary to aggregate by location

user_loc_hastag_data_dic = user_loc_hastag_data.to_dict(orient='records')

print("There are a total of {} records in the dictionary".format(len(user_loc_hastag_data_dic)))
# code block to perform string manipulation and extract location keys and aggregated values

cleaned_dic_container = []

for each in user_loc_hastag_data_dic:

    if str(each["user_location"]).lower() != 'nan' and str(each["hashtags"]).lower() != 'nan':

        cleaned_dic = {}

        each["hashtags"] = str(each["hashtags"]).strip('[]').replace("'","").split(",")

        cleaned_dic["user_location"] = str(each["user_location"])

        cleaned_dic["hashtags"] = each["hashtags"]

        cleaned_dic_container.append(cleaned_dic)

cleaned_dic_container[0:5]
# converting the processed list of dictionaries to a dataframe by using the 'explode' method of pandas to spread each of the 'hashtag' column entries vertically

user_loc_hashtags_df = pd.DataFrame(cleaned_dic_container)

user_loc_hashtags_df = user_loc_hashtags_df.explode('hashtags')

user_loc_hashtags_df
# applying final manipulations using lambda functions

hashtag_loc_df = user_loc_hashtags_df.groupby(['user_location',"hashtags"])["hashtags"].count().reset_index(name="count").sort_values(by=['count'], ascending=False).head(100)

hashtag_loc_df["user_location"] = hashtag_loc_df["user_location"].apply(lambda x : x.strip())

hashtag_loc_df["hashtags"] = hashtag_loc_df["hashtags"].apply(lambda x : x.strip())

hashtag_loc_df
fig = px.bar(hashtag_loc_df,x = "user_location",y="count",color="hashtags",title="What are these countries talking about the most ?")

fig.show()
# grouping by user location , user verified and the source. extracting the top 50 most commonly used sources where the users are verified.

user_loc_source_df = data.groupby(["user_location","user_verified","source"])["source"].count().reset_index(name="count").sort_values(by=['count'], ascending=False).head(50)

user_loc_source_df.head()
fig = px.bar(user_loc_source_df,x="user_location", y="count",color="source", facet_col="user_verified",title="Exploring the relationship between the user location v/s source of the tweet v/s user verification status")

fig.show()
# grouping by user verified status and finding the median user followers

user_ver_followers_df = data.groupby("user_verified")["user_followers"].median().reset_index(name="median_followers").sort_values(by=["median_followers"],ascending=False)

user_ver_followers_df
fig = px.pie(user_ver_followers_df, values="median_followers",names="user_verified",hole=.5, title="The proportion of Median number of followers between user verified accounts v/s unverified accounts")

fig.show()
user_ver_friends_df = data.groupby("user_verified")["user_friends"].median().reset_index(name="median_friends").sort_values(by=["median_friends"],ascending=False)

user_ver_friends_df
fig = px.pie(user_ver_friends_df, values="median_friends",names="user_verified",hole=.5, title="The proportion of Median number of friends between user verified accounts v/s unverified accounts")

fig.show()
user_ver_fav_retweet_df = data.groupby("user_verified",as_index=False).agg({"user_favourites":"median","is_retweet":"count"})

user_ver_fav_retweet_df
fig = px.bar(user_ver_fav_retweet_df, x="is_retweet",y="user_favourites",color="user_verified",orientation='h', 

             title = "Relationship between user_favorites, number of retweets made grouped on user_verified status" )

fig.show()
time_series_data = data.groupby('day_of_tweet',as_index=False).agg({'user_followers':'median','user_verified':'count'})

time_series_data
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=time_series_data["day_of_tweet"], y=time_series_data["user_followers"],

                    mode='lines+markers',

                    name='number of followers'))

fig.add_trace(go.Scatter(x=time_series_data["day_of_tweet"], y=time_series_data["user_verified"],

                    mode='lines+markers',

                    name='number of verified users'))



fig.update_layout(title='Time Series data for change in followers with number of verified users',

                   xaxis_title='Day',

                   yaxis_title='Number of users')
time_series_retweet_df = data.groupby(['day_of_tweet','user_verified'],as_index=False).agg({'is_retweet':'count'})

time_series_retweet_df
fig = px.line(time_series_retweet_df, x="day_of_tweet", y="is_retweet", color='user_verified',title="Time Series representation for change of retweets made on verified and unverified users")

fig.show()