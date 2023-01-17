# Importing Libraries 

import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Files available in our data sets

import os

print(os.listdir("../input"))
# Loading Data

train_df = pd.read_csv("../input/train_tweets.csv")

test_df = pd.read_csv("../input/test_tweets.csv")
#Training Data Set

train_df.head(10)
#Testing Data Set

test_df.head()

print('Testing data set has no Label column')

print(test_df.head(10))
# Training Data Set Information

print("Training Data Set Info - Total Rows | Total Columns | Total Null Values")

print(train_df.info())
# Testing Data Set Information

print("Test Data Set Info - Total Rows | Total Columns | Total Null Values")

print(test_df.info())
# Merging both the data sets as tweets in both the data set is unstructured

combine_df = train_df.append(test_df, ignore_index = True, sort = False)

combine_df.head()
# Combine (Merged) Data Set Information

print("Combine Data Set Info - Total Rows | Total Columns | Total Null Values")

print(combine_df.info())
print("""Step A : Converting html entities i.e. (&lt; &gt; &amp;)

( "&lt;" is converted to “<” and "&amp;" is converted to “&”)""")
# Importing HTMLParser

from html.parser import HTMLParser

html_parser = HTMLParser()
# Created a new columns i.e. clean_tweet contains the same tweets but cleaned version

combine_df['clean_tweet'] = combine_df['tweet'].apply(lambda x: html_parser.unescape(x))

combine_df.head(10)
def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

    return input_txt
# remove twitter handles (@user)

combine_df['clean_tweet'] = np.vectorize(remove_pattern)(combine_df['clean_tweet'], "@[\w]*")

combine_df.head(10)
combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: x.lower())

combine_df.head(10)
# Apostrophe Dictionary

apostrophe_dict = {

"ain't": "am not / are not",

"aren't": "are not / am not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had / he would",

"he'd've": "he would have",

"he'll": "he shall / he will",

"he'll've": "he shall have / he will have",

"he's": "he has / he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how has / how is",

"i'd": "I had / I would",

"i'd've": "I would have",

"i'll": "I shall / I will",

"i'll've": "I shall have / I will have",

"i'm": "I am",

"i've": "I have",

"isn't": "is not",

"it'd": "it had / it would",

"it'd've": "it would have",

"it'll": "it shall / it will",

"it'll've": "it shall have / it will have",

"it's": "it has / it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she had / she would",

"she'd've": "she would have",

"she'll": "she shall / she will",

"she'll've": "she shall have / she will have",

"she's": "she has / she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so as / so is",

"that'd": "that would / that had",

"that'd've": "that would have",

"that's": "that has / that is",

"there'd": "there had / there would",

"there'd've": "there would have",

"there's": "there has / there is",

"they'd": "they had / they would",

"they'd've": "they would have",

"they'll": "they shall / they will",

"they'll've": "they shall have / they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we had / we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what shall / what will",

"what'll've": "what shall have / what will have",

"what're": "what are",

"what's": "what has / what is",

"what've": "what have",

"when's": "when has / when is",

"when've": "when have",

"where'd": "where did",

"where's": "where has / where is",

"where've": "where have",

"who'll": "who shall / who will",

"who'll've": "who shall have / who will have",

"who's": "who has / who is",

"who've": "who have",

"why's": "why has / why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you had / you would",

"you'd've": "you would have",

"you'll": "you shall / you will",

"you'll've": "you shall have / you will have",

"you're": "you are",

"you've": "you have"

}

apostrophe_dict
def lookup_dict(text, dictionary):

    for word in text.split():

        if word.lower() in dictionary:

            if word.lower() in text.split():

                text = text.replace(word, dictionary[word.lower()])

    return text
combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: lookup_dict(x,apostrophe_dict))

combine_df.head(10)
short_word_dict = {

"121": "one to one",

"a/s/l": "age, sex, location",

"adn": "any day now",

"afaik": "as far as I know",

"afk": "away from keyboard",

"aight": "alright",

"alol": "actually laughing out loud",

"b4": "before",

"b4n": "bye for now",

"bak": "back at the keyboard",

"bf": "boyfriend",

"bff": "best friends forever",

"bfn": "bye for now",

"bg": "big grin",

"bta": "but then again",

"btw": "by the way",

"cid": "crying in disgrace",

"cnp": "continued in my next post",

"cp": "chat post",

"cu": "see you",

"cul": "see you later",

"cul8r": "see you later",

"cya": "bye",

"cyo": "see you online",

"dbau": "doing business as usual",

"fud": "fear, uncertainty, and doubt",

"fwiw": "for what it's worth",

"fyi": "for your information",

"g": "grin",

"g2g": "got to go",

"ga": "go ahead",

"gal": "get a life",

"gf": "girlfriend",

"gfn": "gone for now",

"gmbo": "giggling my butt off",

"gmta": "great minds think alike",

"h8": "hate",

"hagn": "have a good night",

"hdop": "help delete online predators",

"hhis": "hanging head in shame",

"iac": "in any case",

"ianal": "I am not a lawyer",

"ic": "I see",

"idk": "I don't know",

"imao": "in my arrogant opinion",

"imnsho": "in my not so humble opinion",

"imo": "in my opinion",

"iow": "in other words",

"ipn": "I’m posting naked",

"irl": "in real life",

"jk": "just kidding",

"l8r": "later",

"ld": "later, dude",

"ldr": "long distance relationship",

"llta": "lots and lots of thunderous applause",

"lmao": "laugh my ass off",

"lmirl": "let's meet in real life",

"lol": "laugh out loud",

"ltr": "longterm relationship",

"lulab": "love you like a brother",

"lulas": "love you like a sister",

"luv": "love",

"m/f": "male or female",

"m8": "mate",

"milf": "mother I would like to fuck",

"oll": "online love",

"omg": "oh my god",

"otoh": "on the other hand",

"pir": "parent in room",

"ppl": "people",

"r": "are",

"rofl": "roll on the floor laughing",

"rpg": "role playing games",

"ru": "are you",

"shid": "slaps head in disgust",

"somy": "sick of me yet",

"sot": "short of time",

"thanx": "thanks",

"thx": "thanks",

"ttyl": "talk to you later",

"u": "you",

"ur": "you are",

"uw": "you’re welcome",

"wb": "welcome back",

"wfm": "works for me",

"wibni": "wouldn't it be nice if",

"wtf": "what the fuck",

"wtg": "way to go",

"wtgp": "want to go private",

"ym": "young man",

"gr8": "great"

}
combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: lookup_dict(x,short_word_dict))

combine_df.head(10)
emoticon_dict = {

":)": "happy",

":‑)": "happy",

":-]": "happy",

":-3": "happy",

":->": "happy",

"8-)": "happy",

":-}": "happy",

":o)": "happy",

":c)": "happy",

":^)": "happy",

"=]": "happy",

"=)": "happy",

"<3": "happy",

":-(": "sad",

":(": "sad",

":c": "sad",

":<": "sad",

":[": "sad",

">:[": "sad",

":{": "sad",

">:(": "sad",

":-c": "sad",

":-< ": "sad",

":-[": "sad",

":-||": "sad"

}

emoticon_dict
combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: lookup_dict(x,emoticon_dict))

combine_df.head(10)
combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))

combine_df.head(10)
combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))

combine_df.head(10)
combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x))

combine_df.head(10)
combine_df['clean_tweet'] = combine_df['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))

combine_df['clean_tweet'][0:5]
from textblob import TextBlob
# Spelling correction is a cool feature which TextBlob offers, we can be accessed using the correct function as shown below.

blob = TextBlob("Why are you stting on this bech??") # Scentence with two errors

print(blob.correct()) # Correct function give us the best possible word simmilar to "gret"
# we can see all the similar matches our first error along with the probability score.

blob.words[3].spellcheck()
# Not cleaning the just showing the spelling check as its take lot of time to process all these tweets

## Shown sample how its must done

text = combine_df['clean_tweet'][0:10].apply(lambda x: str(TextBlob(x).correct()))

text
# Importing stop words from NLTK coupus and word tokenizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
# Creating token for the clean tweets

combine_df['tweet_token'] = combine_df['clean_tweet'].apply(lambda x: word_tokenize(x))



## Fully formated tweets & there tokens

combine_df.head(10)
# Importing stop words from NLTK corpus for english language

stop_words = set(stopwords.words('english'))

stop_words
# Created new columns of tokens - where stop words are being removed

combine_df['tweet_token_filtered'] = combine_df['tweet_token'].apply(lambda x: [word for word in x if not word in stop_words])



## Tokens columns with stop words and without stop words

combine_df[['tweet_token', 'tweet_token_filtered']].head(10)
# Importing library for stemming

from nltk.stem import PorterStemmer

stemming = PorterStemmer()
# Created one more columns tweet_stemmed it shows tweets' stemmed version

combine_df['tweet_stemmed'] = combine_df['tweet_token_filtered'].apply(lambda x: ' '.join([stemming.stem(i) for i in x]))

combine_df['tweet_stemmed'].head(10)
# Importing library for lemmatizing

from nltk.stem.wordnet import WordNetLemmatizer

lemmatizing = WordNetLemmatizer()
# Created one more columns tweet_lemmatized it shows tweets' lemmatized version

combine_df['tweet_lemmatized'] = combine_df['tweet_token_filtered'].apply(lambda x: ' '.join([lemmatizing.lemmatize(i) for i in x]))

combine_df['tweet_lemmatized'].head(10)
# Our final dataframe - Fully formatted, Processed, Noise less, Cleaned, ready to analyse

## for further analysis we consider 2 columns i.e. "tweet_stemmed" & "tweet_lematized"

### We are using 2 columns to see which of them give us better score.

combine_df.head(10)
#visualizing all the words in column "tweet_stemmed" in our data using the wordcloud plot.

all_words = ' '.join([text for text in combine_df['tweet_stemmed']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.title("Most Common words in column Tweet Stemmed")

plt.show()
#Visualizing all the words in column "tweet_lemmatized" in our data using the wordcloud plot.

all_words = ' '.join([text for text in combine_df['tweet_lemmatized']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.title("Most Common words in column Tweet Lemmatized")

plt.show()
#Visualizing all the normal or non racist/sexist words in column "tweet_stemmed" in our data using the wordcloud plot.

normal_words =' '.join([text for text in combine_df['tweet_stemmed'][combine_df['label'] == 0]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.title("Most non racist/sexist words in column Tweet Stemmed")

plt.show()
#Visualizing all the normal or non racist/sexist words in column "tweet_lemmatized" in our data using the wordcloud plot.

normal_words =' '.join([text for text in combine_df['tweet_lemmatized'][combine_df['label'] == 0]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.title("Most non racist/sexist words in column Tweet Lemmatized")

plt.show()
#Visualizing all the negative or racist/sexist words in column "tweet_stemmed" in our data using the wordcloud plot.

negative_words =' '.join([text for text in combine_df['tweet_stemmed'][combine_df['label'] == 1]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.title("Most racist/sexist words in column Tweet Stemmed")

plt.show()
#Visualizing all the negative or racist/sexist words in column "tweet_lemmatized" in our data using the wordcloud plot.

negative_words =' '.join([text for text in combine_df['tweet_lemmatized'][combine_df['label'] == 1]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.title("Most racist/sexist words in column Tweet Lemmatized")

plt.show()
# Importing library

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow_vectorizer
# bag-of-words feature matrix - For columns "combine_df['tweet_stemmed']"

bow_stem = bow_vectorizer.fit_transform(combine_df['tweet_stemmed'])

bow_stem
# bag-of-words feature matrix - For column - combine_df['tweet_lemmatized']

bow_lemm = bow_vectorizer.fit_transform(combine_df['tweet_lemmatized'])

bow_lemm
# Importing library

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

tfidf_vectorizer
# TF-IDF feature matrix - For columns "combine_df['tweet_stemmed']"

tfidf_stem = tfidf_vectorizer.fit_transform(combine_df['tweet_stemmed'])

tfidf_stem
# TF-IDF feature matrix - For columns "combine_df['tweet_lemmatized']"

tfidf_lemm = tfidf_vectorizer.fit_transform(combine_df['tweet_lemmatized'])

tfidf_lemm
# Importing Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
# A.1 For columns "combine_df['tweet_stemmed']"

train_bow = bow_stem[:31962,:]

test_bow = bow_stem[31962:,:]



# splitting data into training and validation set

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train_df['label'], random_state=42, test_size=0.3)



lreg = LogisticRegression()

lreg.fit(xtrain_bow, ytrain) # training the model



prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set

prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0

prediction_int = prediction_int.astype(np.int)



A1 = f1_score(yvalid, prediction_int) # calculating f1 score

print(A1)
# A.2 For columns "combine_df['tweet_lemmatized']"

train_bow = bow_lemm[:31962,:]

test_bow = bow_lemm[31962:,:]



# splitting data into training and validation set

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train_df['label'], random_state=42, test_size=0.3)



lreg = LogisticRegression()

lreg.fit(xtrain_bow, ytrain) # training the model



prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set

prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0

prediction_int = prediction_int.astype(np.int)



A2 = f1_score(yvalid, prediction_int) # calculating f1 score

print(A2)
# B.1 For columns "combine_df['tweet_stemmed']"

train_tfidf = tfidf_stem[:31962,:]

test_tfidf = tfidf_stem[31962:,:]



xtrain_tfidf = train_tfidf[ytrain.index]

xvalid_tfidf = train_tfidf[yvalid.index]



lreg.fit(xtrain_tfidf, ytrain)



prediction = lreg.predict_proba(xvalid_tfidf)

prediction_int = prediction[:,1] >= 0.3

prediction_int = prediction_int.astype(np.int)



B1 = f1_score(yvalid, prediction_int) # calculating f1 score

print(B1)
# B.2 For columns "combine_df['tweet_lemmatized']"

train_tfidf = tfidf_lemm[:31962,:]

test_tfidf = tfidf_lemm[31962:,:]



xtrain_tfidf = train_tfidf[ytrain.index]

xvalid_tfidf = train_tfidf[yvalid.index]



lreg.fit(xtrain_tfidf, ytrain)



prediction = lreg.predict_proba(xvalid_tfidf)

prediction_int = prediction[:,1] >= 0.3

prediction_int = prediction_int.astype(np.int)



B2 = f1_score(yvalid, prediction_int) # calculating f1 score

print(B2)
print("F1 - Score Chart")

print("* F1-Score - Model using Bag-of-Words features")

print("   F1-Score = ",A1," - For column tweets are stemmed")

print("   F1-Score = ",A2," - For column tweets are Lemmatized")

print("* F1-Score - Model using TF-IDF features")

print("   F1-Score = ",B1," - For column tweets are stemmed")

print("   F1-Score = ",B2," - For column tweets are Lemmatized")
