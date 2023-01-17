# download necessary packages

!pip install langdetect

!pip install emoji
# load modules

import pandas as pd

from datetime import datetime, date, timedelta

import numpy as np

import re

import os



import matplotlib.pyplot as plt

import seaborn as sns



import nltk 

nltk.download('stopwords')

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

import gensim

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit,train_test_split, GroupShuffleSplit

from langdetect import detect

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize

from keras.wrappers.scikit_learn import KerasClassifier

from textblob import TextBlob





os.environ['KMP_DUPLICATE_LIB_OK']='True'
# load CAN-NPI dataset

npis_csv = "/kaggle/input/covid19-challenges/npi_canada.csv"

raw_data = pd.read_csv(npis_csv,encoding = "ISO-8859-1")

# remove any rows that don't have a start_date, region, or intervention_category

df = raw_data.dropna(how='any', subset=['start_date', 'region', 'intervention_category'])

df['region'] = df['region'].replace('Newfoundland', 'Newfoundland and Labrador')

num_rows_removed = len(raw_data)-len(df)

print("Number of rows removed: {}".format(num_rows_removed))



# get all regions

regions = list(set(df.region.values))

print("Number of unique regions: {}".format(len(regions)))



# get all intervention categories

num_cats = list(set(df.intervention_category.values))

num_interventions = len(num_cats)

print("Number of unique intervention categories: {}".format(len(num_cats)))



# get earliest start date and latest start date

df['start_date'] = pd.to_datetime(df['start_date'], format='%Y-%m-%d')

earliest_start_date = df['start_date'].min()

latest_start_date = df['start_date'].max()

num_days = latest_start_date - earliest_start_date

print("Analyzing from {} to {} ({} days)".format(earliest_start_date.date(), latest_start_date.date(), num_days))

print("DONE READING DATA")
# load tweets

merged_tweets_csv = '/kaggle/input/npi-twitterverse-april-30/tweets_to_intervention_category.source_urls.tsv'

colnames = ["npi_record_id", "intervention_category", "oxford_government_response_category", "source_url", "id", "conversation_id", "created_at", "date", "time", "timezone", "user_id", "username", "name", "place", "tweet", "mentions", "urls", "photos", "replies_count", "retweets_count", "likes_count", "hashtags", "cashtags", "link", "retweet", "quote_url", "video", "near", "geo", "source", "user_rt_id", "user_rt", "retweet_id", "reply_to", "retweet_date", "translate", "trans_src", "trans_dest"]

tweets_df = pd.read_csv(merged_tweets_csv, encoding = "utf-8", error_bad_lines=False, engine='python', names=colnames)

# drop any rows without tweets - aka any interventions supported by non-tweeted media urls

tweets_df = tweets_df.dropna(how='any', subset=['npi_record_id', 'intervention_category', 'tweet'])



# only get english tweets

data = []

for index, row in tweets_df.iterrows():

    # detect only english tweets

    tweet = row['tweet'].strip()

    if tweet != "":

        language =""

        try:

            language = detect(tweet)

        except:

            language = "error"

        if language == "en":

            data.append([row['intervention_category'], tweet])

tweets_df_en = pd.DataFrame(data, columns=["intervention_category", "tweet"])

print("Number of non-english tweets = {}".format(len(tweets_df) - len(tweets_df_en)))

print("Number of tweets collected = {}".format(len(tweets_df_en)))
# Here's a few examples of First death announcements

ex1 = "Here's a wrap of the latest coronavirus news in Canada: 77 cases, one death, an outbreak in a B.C. nursing home and Ottawa asks provinces about their critical supply gaps.  https://www.theglobeandmail.com/canada/article-bc-records-canadas-first-coronavirus-death/"

ex2 = "B.C. records Canada’s first coronavirus death  http://dlvr.it/RRZPGL  pic.twitter.com/pn8T4yumQJ"

print("Example 1 = {}".format(ex1))

print("Example 2 = {}".format(ex2))
ex1_tb = TextBlob(ex1)

ex1_ss = ex1_tb.sentiment[0]

print("Example 1 has score={}".format(ex1_ss))

ex2_tb = TextBlob(ex2)

ex2_ss = ex2_tb.sentiment[0]

print("Example 2 has score={}".format(ex2_ss))
ex = "first coronavirus death"

ex_tb = TextBlob(ex)

ex_ss = ex_tb.sentiment[0]

print("{} with score={}".format(ex, ex_ss))



ex = "coronavirus death"

ex_tb = TextBlob(ex)

ex_ss = ex_tb.sentiment[0]

print("{} with score={}".format(ex, ex_ss))
import re 

import nltk

nltk.download('punkt')



def tweet_preprocess(text):

  '''Return tokenized text with 

  rsemoved URLs, usernames, hashtags, weird characters, repeated

  characters, stop words, and numbers

  '''

  text = text.lower()

  text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text) # remove URLs

  text = re.sub(r'@[A-Za-z0-9]+','USER',text) # removes any usernames in tweets

  text = re.sub(r'#([^\s]+)', r'\1', text) # remove the # in #hashtag

  text = re.sub('[^a-zA-Z0-9-*. ]', ' ', text) # remove any remaining weird characters

  words = word_tokenize(text)  # remove repeated characters (helloooooooo into hello)

  ignore = set(stopwords.words('english'))

  more_ignore = {'at', 'and', 'also', 'or', "http", "ca", "www", "https", "com", "twitter", "html", "news", "link", \

                 "positive", "first", "First", "confirmed", "confirm", "confirms"}

  ignore.update(more_ignore)

  #porter = PorterStemmer()

  #cleaned_words_tokens = [porter.stem(w) for w in words if w not in ignore]

  cleaned_words_tokens = [w for w in words if w not in ignore]

  cleaned_words_tokens = [w for w in cleaned_words_tokens if w.isalpha()]



  return cleaned_words_tokens
def run_sentiment_analysis(tweets_df):

  tweets_df["sentiment"] = 0

  for index, row in tweets_df.iterrows():

    tokens = tweet_preprocess(row['tweet'])

    clean_text = ' '.join(tokens)

    analysis = TextBlob(row['tweet'])

    analysis_after_clean = TextBlob(clean_text)



    print("{}: {} \n before cleaning score={}, after cleaning score={}".format(row['intervention_category'], row['tweet'], analysis.sentiment[0], analysis_after_clean.sentiment[0]))



    if analysis.sentiment[0]>0:

      print('Positive')

    elif analysis.sentiment[0]<0:

      print('Negative')

    else:

      print('Neutral')

    print("======================================")
run_sentiment_analysis(tweets_df_en[:5])
# download sentiment map

!wget https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1048/Emoji_Sentiment_Data_v1.0.csv
import emoji



# get emoji sentiment map

emoji_sent_csv = "Emoji_Sentiment_Data_v1.0.csv"

emoji_data = pd.read_csv(emoji_sent_csv,encoding = "ISO-8859-1")



def extract_emojis(str):

  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)



def calc_emoji_sent(e):

    e_uc = '0x{:X}'.format(ord(e)).lower()

    #print(e_uc)

    count_pos =0

    count_neg =0

    count_neutral = 0

    sr = emoji_data.loc[emoji_data["Unicode codepoint"] == e_uc.lower()]

    score = -100

    if not sr.empty:

        oc = int(sr["Occurrences"].astype(int))

        num_pos = int(sr["Positive"].astype(int))

        num_neut = int(sr["Neutral"].astype(int))

        num_neg = int(sr["Negative"].astype(int))

        score = 1*num_pos/oc + -1*num_neg/oc + 0*num_neut/oc

    #print("{} with score={}".format(e, score))

    return score



def run_sentiment_analysis_mod(tweets_df):

  tweets_df["sentiment_score"] = 0.0

  tweets_df["sentiment_class"] = ""



  for index, row in tweets_df.iterrows():

    tokens = tweet_preprocess(row['tweet'])

    clean_text = ' '.join(tokens)

    analysis = TextBlob(row['tweet'])

    analysis_after_clean = TextBlob(clean_text)

    c_score = analysis_after_clean.sentiment[0]

    

    # add emojis in sentiment analysis

    emojis_detected = extract_emojis(row['tweet'])

    avg_emoji_sent_score = 0

    emoji_counts = 0

    if emojis_detected:

        for e in emojis_detected:

            em_sent_score = calc_emoji_sent(e)

            if em_sent_score == -100:

              continue

            avg_emoji_sent_score += em_sent_score

            emoji_counts += 1

        if emoji_counts > 0:

            avg_emoji_sent_score = avg_emoji_sent_score/emoji_counts

        #print(avg_emoji_sent_score)





    # final score calculations

    score = 0.0

    label = "NEUTRAL"

    if avg_emoji_sent_score > 0.10:

        score = avg_emoji_sent_score

        label = "POSITIVE"

    elif avg_emoji_sent_score < -0.10:

        score = avg_emoji_sent_score

        label = "NEGATIVE"

    else:

        score = analysis_after_clean.sentiment[0]

        if score > 0.25:

          label = "POSITIVE"

        elif score < -0.25:

          label = "NEGATIVE"

    tweets_df.at[index, "sentiment_score"] = score

    tweets_df.at[index, "sentiment_class"] = label 

    '''print("=============================")

    print(row["intervention_category"] + "\n")

    print(row['tweet'])

    print(clean_text)

    print("Score (no clean) = {}".format(analysis.sentiment[0]))

    print("Score (clean) = {}".format(c_score))

    print("Final Score = {}".format(score))

    print(label)'''

  return tweets_df



mod_tweets_df = run_sentiment_analysis_mod(tweets_df)
import plotly.graph_objects as go

import plotly



def split_data_by_class(tweets_df):

    total_tweets_by_cat = tweets_df.groupby('intervention_category')["id"].count().reset_index(name="count").sort_values("intervention_category", ascending=False)

    counts = tweets_df.groupby(['intervention_category',"sentiment_class"])["id"].count().reset_index(name="count").sort_values("intervention_category", ascending=False)

    counts["proportion"] = 0.0

    for index, row in counts.iterrows():

        total_tweets = int(total_tweets_by_cat.loc[total_tweets_by_cat["intervention_category"] == row["intervention_category"]]["count"].astype(int))

        counts.at[index, "proportion"] = row["count"]/total_tweets



    y = counts["intervention_category"].unique().tolist()



    # fill gaps - some sentiment_class + intervention_category combinations are empty

    # and it messes up my graphs :(

    fill_data = []

    for ic in y:

      for sc in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:

        subset = counts[(counts.sentiment_class == sc) & (counts.intervention_category == ic)]

        if subset.empty:

          fill_data.append([ic, sc, 0, 0.0])

    fill_data_df = pd.DataFrame(fill_data, columns=["intervention_category", "sentiment_class", "count", "proportion"])

    full_counts = counts.append(fill_data_df).sort_values("intervention_category", ascending=False)



    return full_counts, y



def plot(full_counts, y, measure):

    # only plot intervention_category if it had "sufficient" number of tweets

    THRESH = 50

    total_tweets_by_cat = tweets_df.groupby('intervention_category')["id"].count().reset_index(name="count").sort_values("intervention_category", ascending=False)

    if measure == "proportion":

      # find all intervention_category with enough tweets

      y = total_tweets_by_cat[total_tweets_by_cat["count"] > THRESH]["intervention_category"].unique().tolist()

      full_counts = full_counts[full_counts.intervention_category.isin(y)]



    # split up by sentiment_class

    pos_counts = full_counts.loc[full_counts["sentiment_class"] == "POSITIVE"]

    neg_counts = full_counts.loc[full_counts["sentiment_class"] == "NEGATIVE"]

    neut_counts = full_counts.loc[full_counts["sentiment_class"] == "NEUTRAL"]

    print("Mean {} for positive class: {}".format(measure, round(pos_counts[measure].mean(),2)))

    print("Mean {} for negative class: {}".format(measure, round(neg_counts[measure].mean(),2)))

    print("Range {} for positive class: {}-{}".format(measure, round(pos_counts[measure].min(),2), round(pos_counts[measure].max(),2)))

    print("Range {}  for negative class: {}-{}".format(measure, round(neg_counts[measure].min(),2), round(neg_counts[measure].max(),2)))

    

    fig = go.Figure()

    fig.add_trace(go.Bar(

        y=y,

        x=pos_counts[measure],

        name='Positive',

        orientation='h',

        marker=dict(

            color='rgba(90, 191,165, 1.0)',

            line=dict(color='rgba(255, 255, 255, 1.0)', width=1)

        )

    ))

    fig.add_trace(go.Bar(

        y=y,

        x=neg_counts[measure],

        name='Negative',

        orientation='h',

        marker=dict(

            color='rgba(230, 130, 130, 1.0)',

            line=dict(color='rgba(255, 255, 255, 1.0)', width=1)

        )

    ))

    fig.add_trace(go.Bar(

        y=y,

        x=neut_counts[measure],

        name='Neutral',

        orientation='h',

        marker=dict(

            color='rgba(190, 203, 200, 1.0)',

            line=dict(color='rgba(255, 255, 255, 1.0)', width=1)

        )

    ))





    fig.update_layout(width=800, height=1200,barmode='stack', 

                      template='plotly_white',

                      bargap=0.5, # gap between bars of adjacent location coordinates.

                      #bargroupgap=0.5 # gap between bars of the same location coordinate.

                     )

    fig.show()

    #plotly.offline.iplot(fig, filename='fig.png')



full_counts, y = split_data_by_class(mod_tweets_df)

plot(full_counts,y, "proportion")

plot(full_counts,y, "count")