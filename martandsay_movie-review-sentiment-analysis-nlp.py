import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import spacy

import nltk

nlp = spacy.load("en_core_web_lg") # Loading english large corpus

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_raw = pd.read_csv("/kaggle/input/imdb-movie-review-dataset/movie_data.csv")
df_raw.head()
df_raw.info()
df_raw.isnull().sum() # We do not have null values
df_counts = df_raw["sentiment"].value_counts().reset_index()

df_counts.head()

df_counts["index"] = df_counts["index"].apply(lambda x : 'Positive' if x == 1 else 'Negative' )

df_counts.head()
# So we can say that we almost have same number of reviews. That mean we have very good data.

plt.figure(figsize=(10, 7))

sns.barplot(data=df_counts, x='index', y='sentiment')

plt.xlabel("Sentiment Type");

plt.ylabel("Total Count");

plt.title("Total Postive & Negative Reviews");
# There may be a case where reviews are not null but empty/blank. Lets check for that

#for i, lb, rv in df_raw.itertuples()[0:10]:

#    print(i, lb, rv)
empty_review_index = []

for i, review, sentiment in df_raw.itertuples():

    # if review type is string

    if type(review) == str:

        #if review is empty space

        if review.isspace():

            # Appent its index to the list

            empty_review_index.append(i)
empty_review_index # So we do not have any empty review
import re # for regular expression
pos_token = [] # to save positive tokens

neg_token = [] # to save negative tokens

corpus=[]

noun = []

def process_reviews(df):

    for index, reviews, sentiment in df.itertuples():

        if type(reviews) == str:

            reviews = re.sub('[^a-zA-Z]', ' ', reviews)

            reviews = reviews.lower()

            doc = nlp(reviews)

            temp = []

            for token in doc:

                if not token.text.isspace():

                    if not token.is_stop and len(token.text) > 2:

                        if token.pos_ == 'NOUN':

                            noun.append(token.text)

                        if sentiment == 1:

                            pos_token.append(token.text)

                        else:

                           

                            neg_token.append(token.text)

                        temp.append(token.lemma_)

                        corpus.append(' '.join(temp))

            
#df_test = df_raw.head(20)

#df_test.head()
process_reviews(df_raw)
# all the positive token list

pos_token[0:10]
# all the negative token list

neg_token[0:10]
from nltk import FreqDist
# preparing frequency distribution variables

freq_pos = FreqDist(pos_token)

freq_neg = FreqDist(neg_token)
freq_pos
freq_neg
# Top 20 most repeated words in positive comments

plt.figure(figsize=(15, 10))

freq_pos.plot(20)
# top 20 most repeated words in negative comments

plt.figure(figsize=(15, 10))

freq_neg.plot(20)
# All the nouns used in our reviews. It will give you an idea like what are the famous keywords?

noun[0:10]
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Most famous nouns used in movie reviews



wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(' '.join(noun))

plt.figure(figsize=(12, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()

# import library for sntiment analysis.

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sm = SentimentIntensityAnalyzer()
# polarit_Scores acuallly gives one dictionary containing 4 valus. negative, positive, neutral and overall value

# compound. So you can see that below sentence has 0.4 neutral and 0.6 negative and overall -0.6696.

# - sign shows it is a negatvie sentence.

sm.polarity_scores("you are so bad")
# Lets try some more sentences. SO below you can see it contains both negative and positive feedback.

# so nltk is smart enough to undertand it. It actually calculate the score of each word and finally calulates

print(sm.polarity_scores("Star wars is amazing. But the the picturization is not good."))
# Now lets see a wierd thing here. I wrote the same sentense as above but score is different. can you say why?

# Actually if you observer i have capitalize the word AMAZING. So here nltk understands the we want to focus

# on word amazing. That is why below sentence is more positive than the above one. Lets try to capitalize

# more words.

print(sm.polarity_scores("Star wars is AMAZING. But the the picturization is not good."))
# Lets apply sentiment analysis on our reviews.
df_raw.head()
df_raw["score"] = df_raw["review"].apply(lambda review : sm.polarity_scores(review))
# extract only compound score

df_raw["sentiment_score"] = df_raw["score"].apply(lambda x: x["compound"]) 
df_raw.head()
# now lets change our sentiment_score column to binary value 0 or 1.

df_raw["sentiment_score"] = df_raw["sentiment_score"].apply(lambda x: 'pos' if x >= 0 else 'neg') 
# now lets change our sentiment_score column to binary value 0 or 1.

df_raw["sentiment"] = df_raw["sentiment"].apply(lambda x: 'pos' if x == 1 else 'neg') 
df_raw.head()
from sklearn.metrics import confusion_matrix, accuracy_score
accuracy_score(df_raw["sentiment"], df_raw["sentiment_score"])
# so we can see we have an accuracy score of 69% which is good in sentiment analysis. Sentiment analysis is not 

# very easy for most of the models as you can not predict srcasm in text. Many people write review as sarcasm,

# which is very difficult or impossible to predict. example:



sm.polarity_scores('Yaaa.. You said it was a good movie... :/')
# In above example i wrote a sarcasm followed by a crap face text smiley but for computer program its just a 

# special characted. So here it doesnt understand the sarcasm. That is why because of these kind of exceptions

# 69% accuracy will be considered as fine number.