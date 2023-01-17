import numpy as np 

import pandas as pd 

from datetime import datetime as dt

import os



# LOAD CSV FILE TO PANDAS DATAFRAME

df = pd.read_csv("../input/the-duck-tweets/tweets_012019_to_062020_translated.csv", error_bad_lines=False)

# 'TweetDate' CONVERT DATETIME TO DATE ONLY

df['TweetDate'] = pd.to_datetime(df['TweetDate'], errors='coerce')

df['TweetDate'] = df['TweetDate'].dt.normalize()
# LINE CHART TO VIEW THE NUMBER OF TWEETS PER DAY

import matplotlib.pyplot as plt



df1 = df.groupby(['TweetDate']).count()



plt.figure(figsize=(30, 10), dpi= 200)

plt.plot(df1)

plt.xticks(rotation=90)

plt.title('Number of Tweets Per Day',fontsize= 30)

plt.xlabel("Date",fontsize= 20)

plt.ylabel("Number of Tweets",fontsize= 20)

plt.legend(fontsize=15)

plt.show()
import nltk

import re

nltk.download('stopwords')

nltk.download('punkt')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



def extract_words(setence):

    text = re.sub("[^a-zA-Z]", " ", setence)

    return(text.lower())



def remove_stopword(sentence):

    new_line = []

    word_tokens = nltk.word_tokenize(sentence)

    for w in word_tokens:

        if w not in stop_words:

            new_line.append(w)

    

    return (" ".join(new_line) + " ").strip()



def get_adjective_word(setence):

    words = []

    pos_nltk = ['JJ','JJR', 'JJS','RB','RBS','RBR']



    nltk_words = nltk.word_tokenize(setence)

    for stc in nltk_words:

        nltk_token = nltk.pos_tag(nltk.word_tokenize(stc))

        if nltk_token[0][1] in pos_nltk and nltk_token[0][0] not in words:

            words.append(nltk_token[0][0])

    

    return (" ".join(words) + " ").strip()



def clean_word(sentence):

    text = re.sub(r'\|\|\|', r' ', sentence) 

    text = re.sub(r'http\S+', r'<URL>', text)

    text = text.replace('x', '')

    return text





for index, row in df.iterrows():

    selected_tweet = row['TweetText'].strip()

    selected_tweet = extract_words(selected_tweet)

    selected_tweet = clean_word(selected_tweet)

    selected_tweet = remove_stopword(selected_tweet)

    selected_tweet = get_adjective_word(selected_tweet)

    df.loc[index, "processed_text"] = selected_tweet 



print(df.head)
!pip install afinn



from afinn import Afinn

afinn = Afinn()



for index, row in df.iterrows():

    selected_tweet = row['TweetText'].strip()

    df.loc[index, "sentiment_scope"] = afinn.score(selected_tweet)

    

# GROUP INTO 3 GROUP ONLY

df.loc[df['sentiment_scope'] < 0, 'sentiment_scope'] = -1

df.loc[df['sentiment_scope'] > 0, 'sentiment_scope'] = 1



print(df.head)
def datasetBySentiment(sentiment_score):

    obj = df[df['sentiment_scope']==sentiment_score]

    obj["TweetYear"] = obj.TweetDate.dt.year

    obj["TweetMonth"] = obj.TweetDate.dt.month

    obj = obj.groupby(["TweetYear", "TweetMonth"], as_index=False).count()

    obj.columns = ['TweetYear', 'TweetMonth', 'TweetCount', 'TBR1', 'TBR2', 'TBR3']

    obj.drop(['TBR1', 'TBR2', 'TBR3'], axis=1, inplace=True)

    obj['TweetPeriod'] = obj['TweetYear'].astype(str) + '-' + obj['TweetMonth'].astype(str)

    obj.drop(['TweetYear', 'TweetMonth'], axis=1, inplace=True)

    return obj



pos_obj = datasetBySentiment(1)

neg_obj = datasetBySentiment(-1)

neu_obj = datasetBySentiment(0)





plt.figure(figsize=(20, 10), dpi= 200)

plt.plot(pos_obj.TweetPeriod, pos_obj.TweetCount, marker='x', color='green', linewidth=2, label="# Positive Sentiment Tweets")

plt.plot(neg_obj.TweetPeriod, neg_obj.TweetCount, marker='+', color='red', linewidth=2, label="# Negative Sentiment Tweets")

plt.plot(neu_obj.TweetPeriod, neu_obj.TweetCount, marker='*', color='grey', linewidth=1, label="# Neutral  Sentiment Tweets")

plt.xticks(rotation=90)

plt.title('Number of Tweets Per Day',fontsize= 20)

plt.xlabel("Date",fontsize= 15)

plt.ylabel("Number of Tweets",fontsize= 15)

plt.legend(loc="upper left")



plt.show()

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

import numpy as np



def create_wordcloud(text, no_of_words):



    wordcloud = WordCloud(background_color="white", max_words=no_of_words, width=800, height=500).generate(text)

    plt.figure(figsize=(8, 12), dpi= 150)

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()



def generate_word(df, columnname, min_count):

    word_list = []

    for index, row in df.iterrows():

        word_list.append(df[columnname][index])



    # PREPARE TEXT

    text = ' '.join(word_list)



    # GENERATE WORDCLOUD

    create_wordcloud(text, min_count)





# POSITIVE

generate_word(df[df['sentiment_scope']>0], "TweetText",100) 



# NEUTRAL

generate_word(df[df['sentiment_scope']==0], "TweetText",100)



# NEGATIVE

generate_word(df[df['sentiment_scope']<0], "TweetText",100)