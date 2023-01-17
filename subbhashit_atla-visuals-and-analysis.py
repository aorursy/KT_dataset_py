import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
series=pd.read_csv("/kaggle/input/avatar-the-last-air-bender/series_names.csv")
series.head()
series.halfway.plot(kind="pie",figsize=(10,10),autopct="%1.1f%%",explode=(0,0.05,0),shadow=True)
data=pd.read_csv("../input/avatar-the-last-air-bender/avatar_data.csv")
data.head()
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data.book=lab.fit_transform(data.book)
import seaborn as sns
sns.jointplot(data.book,data.imdb_rating,kind="kde")
plt.figure(figsize=(15,9))
sns.lineplot(data.imdb_rating, data.series_rating, color="g")
scenes=pd.read_csv("../input/avatar-the-last-air-bender/scenes.csv", encoding= 'unicode_escape')
scenes.head()
def sentence(x):
    a=x.split('[')
    b=a[1].split(']')
    return b[0]
scenes["scene_description"]=scenes["scene_description"].apply(sentence)
scenes.head()
import re
def process(x):
    processed_tweet = re.sub(r'\W', ' ', str(x))
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    processed_tweet = processed_tweet.lower()
    return processed_tweet
scenes.scene_description=scenes.scene_description.apply(process)
import nltk
from nltk.stem import PorterStemmer,LancasterStemmer
stemming =PorterStemmer()
def identify_tokens(row):
    tokens = nltk.word_tokenize(row)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words
def stem_list(row):
    stemmed_list = [stemming.stem(word) for word in row]
    return (stemmed_list)
def rejoin_words(row):
    joined_words = ( " ".join(row))
    return joined_words
scenes.scene_description=scenes.scene_description.apply(identify_tokens)
scenes.scene_description=scenes.scene_description.apply(stem_list)
scenes.scene_description=scenes.scene_description.apply(rejoin_words)
from textblob import TextBlob
scenes.scene_description=scenes.scene_description.apply(lambda x:TextBlob(x).sentiment.subjectivity)
scenes.head()