import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from collections import Counter
from wordcloud import WordCloud
from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier


from keras import models
from keras import layers
import keras
from keras import optimizers
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
hp = pd.read_csv('/kaggle/input/harry-potter-and-the-philosophers-stone-script/hp_script.csv',encoding='cp1252')
hp.head()
hp['character_name'].value_counts()
sns.set_style('whitegrid')
plt.figure(figsize=(10,7))
sns.countplot(y='character_name', data=hp, order=hp.character_name.value_counts().iloc[:20].index, palette="Reds_d")
plt.xlabel('Number of lines of dialogue', fontsize=15)
plt.ylabel('Character', fontsize=15)
plt.title('Character Importance by Number of Lines of Dialogue', fontsize=20)
plt.show()
# adding a new column to the dataframe, of number of words in each line
hp['dialogue_wordcount'] = hp['dialogue'].map(lambda x:len(re.findall(r'\w+', x)))
hp
total_char_words = hp.groupby('character_name', as_index=False).dialogue_wordcount.sum()
total_char_words = pd.DataFrame(total_char_words)
total_char_words
sns.set_style('whitegrid')
plt.figure(figsize=(10,7))
sns.barplot(x='dialogue_wordcount',y='character_name', data=total_char_words, palette="Purples_d", order=total_char_words.sort_values('dialogue_wordcount', ascending=False).character_name[0:20], orient='h')
plt.xlabel('Number of words of dialogue', fontsize=15)
plt.ylabel('Character', fontsize=15)
plt.title('Character Importance by Number of Words of Dialogue', fontsize=20)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words = [PorterStemmer().stem(w) for w in filtered_words]
    lemma_words=[WordNetLemmatizer().lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)
hp['cleanText']=hp['dialogue'].map(lambda x:preprocess(x))
common_words = Counter(" ".join(hp["cleanText"]).split()).most_common(10)
common_words
text = " ".join(line for line in hp["cleanText"])
wordcloud = WordCloud(width=1000, height=1000, background_color="white", min_font_size=15).generate(text)
plt.figure(figsize = (10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
harry = hp[hp['character_name']=='Harry Potter']
common_harry = Counter(" ".join(harry["cleanText"]).split()).most_common(5)
common_harry
harry_text = " ".join(line for line in harry["cleanText"])
wordcloud = WordCloud(width=1000, height=1000, background_color="white", min_font_size=15).generate(harry_text)
plt.figure(figsize = (10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
ron = hp[hp['character_name']=='Ron Weasley']
common_ron = Counter(" ".join(ron["cleanText"]).split()).most_common(5)
common_ron
ron_text = " ".join(line for line in ron["cleanText"])
wordcloud = WordCloud(width=1000, height=1000, background_color="white", min_font_size=15).generate(ron_text)
plt.figure(figsize = (10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
hermione = hp[hp['character_name']=='Hermione Granger']
common_hermione = Counter(" ".join(hermione["cleanText"]).split()).most_common(5)
common_hermione
hermione_text = " ".join(line for line in hermione["cleanText"])
wordcloud = WordCloud(width=1000, height=1000, background_color="white", min_font_size=15).generate(hermione_text)
plt.figure(figsize = (10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# loading the data and adjusting to provide column names

twitter = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',encoding='cp1252', names = ['label', 'id', 'date', 'flag', 'user', 'text'])
twitter.head()
# dropping irrelevant columns

twitter = twitter.drop(['id', 'date', 'flag', 'user'], axis=1)
# Dataset contains 1,600,000 rows. This is a very large dataset, so I will take a sample 1/4 the size of this.

twit_samp = twitter.sample(n=400000,replace=False)
# creating new column of clean text using previously defined function

twit_samp['cleanText']=twit_samp['text'].map(lambda x:preprocess(x))
# filtering data to only use tweets with more than two words after processing

twit_samp['clean_wordcount'] = twit_samp['cleanText'].map(lambda x:len(re.findall(r'\w+', x)))
filtered_twit = twit_samp[twit_samp['clean_wordcount'] > 2]
x_train_samp = filtered_twit['cleanText']
y_train_samp = filtered_twit['label']
y_train_samp = y_train_samp.replace(4,1)
# creating training and validation sets - 90% training, 10% validation

x_train_samp, x_valid_samp, y_train_samp, y_valid_samp = train_test_split(x_train_samp, y_train_samp, test_size=0.1)
# this converts the words into vectors of numbers to allow use within models

tokenizer = RegexpTokenizer(r'\w+')
vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)
full_text = list(x_train_samp.values) + list(x_valid_samp.values)
vectorizer.fit(full_text)
train_vectorized_samp = vectorizer.transform(x_train_samp)
test_vectorized_samp = vectorizer.transform(x_valid_samp)
hp_vectorized = vectorizer.transform(hp['cleanText'])
logreg = LogisticRegression(max_iter=1000, multi_class='multinomial')
logreg.fit(train_vectorized_samp, y_train_samp)
logreg.score(test_vectorized_samp, y_valid_samp)
linsvc = LinearSVC(max_iter=2000)
linsvc.fit(train_vectorized_samp, y_train_samp)
linsvc.score(test_vectorized_samp, y_valid_samp)
multinb = MultinomialNB()
multinb.fit(train_vectorized_samp, y_train_samp)
multinb.score(test_vectorized_samp, y_valid_samp)
bernb = BernoulliNB()
bernb.fit(train_vectorized_samp, y_train_samp)
bernb.score(test_vectorized_samp, y_valid_samp)
preds = logreg.predict(hp_vectorized)
hp['sentiment_preds'] = preds
hp.head()
hp['clean_wordcount'] = hp['cleanText'].map(lambda x:len(re.findall(r'\w+', x)))
filtered_hp = hp[hp['clean_wordcount'] > 2]
filtered_hp.head()
# to filter the data further, I will only analyse the top 25 characters with the most lines

char_counts = filtered_hp['character_name'].value_counts()
char_counts = char_counts[0:25]
char_counts = pd.DataFrame(char_counts)
char_counts['count'] = char_counts['character_name']
char_counts['character_name'] = char_counts.index
char_counts = char_counts.reset_index()
char_counts.drop('index', axis=1)
filtered_hp = filtered_hp[filtered_hp.character_name.isin(char_counts['character_name'])]
filtered_hp.head()
pos_neg_chars = filtered_hp.groupby('character_name', as_index=False).sentiment_preds.mean()
pos_neg_chars = pd.DataFrame(pos_neg_chars)
pos_neg_chars
plt.figure(figsize=(10,7))
sns.barplot(x='sentiment_preds', y='character_name', data=pos_neg_chars, palette="Greens_d", order=pos_neg_chars.sort_values('sentiment_preds', ascending=False).character_name[0:25], orient='h')
plt.xlabel('Positivity', fontsize=15)
plt.ylabel('Character', fontsize=15)
plt.title('Mean Sentiment of Top 25 Characters', fontsize=20)
plt.show()
scene_sent = pd.DataFrame(hp.groupby('scene', as_index=False).sentiment_preds.mean())
plt.figure(figsize=(10,7))
sns.lineplot(x="scene", y="sentiment_preds", data=scene_sent)
plt.xlabel('Scene Number', fontsize=15)
plt.ylabel('Mean Sentiment', fontsize=15)
plt.title('Mean Sentiment Progression Throughout Movie', fontsize=20)
plt.ylim(0,1)
plt.xlim(1,34)
plt.show()
emotions = pd.read_csv('/kaggle/input/emotion/text_emotion.csv')
emotions = emotions.drop(columns = ['tweet_id', 'author'])
emotions.groupby('sentiment').count()
# due to the very small number of anger and boredom tweets in comparison to the other emotions, I will remove these tweets

emotions = emotions[emotions.sentiment.isin({'empty', 'enthusiasm', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry'})]
# converting each emotion to an integer

emotions['sentiment'] = emotions['sentiment'].map({'empty':0, 'enthusiasm':1, 'fun':2, 'happiness':3, 'hate':4, 'love':5, 'neutral':6, 'relief':7, 'sadness':8, 'surprise':9, 'worry':10})
emotions.head()
emotions['cleanText']=emotions['content'].map(lambda x:preprocess(x))
x_train2 = emotions['cleanText']
y_train2 = emotions['sentiment']
x_vectorized = vectorizer.transform(x_train2)
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x_vectorized, y_train2)
train2_vectorized, test2_vectorized, y_train2, y_valid2 = train_test_split(x_smote, y_smote, test_size=0.1)
logreg2 = LogisticRegression(max_iter=500)
logreg2.fit(train2_vectorized, y_train2)
logreg2.score(test2_vectorized, y_valid2)
linsvc2 = LinearSVC(max_iter=800)
linsvc2.fit(train2_vectorized, y_train2)
linsvc2.score(test2_vectorized, y_valid2)
multinb2 = MultinomialNB()
multinb2.fit(train2_vectorized, y_train2)
multinb2.score(test2_vectorized, y_valid2)
preds2 = linsvc2.predict(hp_vectorized)
hp['emotion_preds'] = preds2
hp['emotion_preds'] = hp['emotion_preds'].map({0:'empty', 1:'enthusiasm', 2:'fun', 3:'happiness', 4:'hate', 5:'love', 6:'neutral', 7:'relief', 8:'sadness', 9:'surprise', 10:'worry'})
total_emotions = pd.DataFrame(hp.groupby('emotion_preds', as_index=False).ID_number.count())
total_emotions = total_emotions.sort_values('ID_number', ascending=False)
plt.figure(figsize=(10,7))
sns.barplot(y='emotion_preds', x='ID_number', data=total_emotions, palette="Oranges_d", orient='h')
plt.title('Most Common Emotions in Entire Movie', fontsize=20)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Emotion', fontsize=15)
char_emotions = pd.DataFrame(hp.groupby('character_name').emotion_preds.value_counts())
char_emotions = char_emotions.rename(columns={'emotion_preds': 'counts'})
char_emotions = char_emotions.reset_index()
plt.figure(figsize=(10,7))
sns.barplot(y='emotion_preds', x='counts', data=char_emotions[char_emotions['character_name']=='Harry Potter'], palette='pink_d')
plt.title('Harry Potter Most Common Emotions', fontsize=20)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Emotion', fontsize=15)
char_emotions = pd.DataFrame(hp.groupby('character_name').emotion_preds.value_counts())
char_emotions = char_emotions.rename(columns={'emotion_preds': 'counts'})
char_emotions = char_emotions.reset_index()
plt.figure(figsize=(10,7))
sns.barplot(y='emotion_preds', x='counts', data=char_emotions[char_emotions['character_name']=='Ron Weasley'], palette='pink_d')
plt.title('Ron Weasley Most Common Emotions', fontsize=20)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Emotion', fontsize=15)
char_emotions = pd.DataFrame(hp.groupby('character_name').emotion_preds.value_counts())
char_emotions = char_emotions.rename(columns={'emotion_preds': 'counts'})
char_emotions = char_emotions.reset_index()
plt.figure(figsize=(10,7))
sns.barplot(y='emotion_preds', x='counts', data=char_emotions[char_emotions['character_name']=='Hermione Granger'], palette='pink_d')
plt.title('Hermione Granger Most Common Emotions', fontsize=20)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Emotion', fontsize=15)
char_emotions = pd.DataFrame(hp.groupby('character_name').emotion_preds.value_counts())
char_emotions = char_emotions.rename(columns={'emotion_preds': 'counts'})
char_emotions = char_emotions.reset_index()
plt.figure(figsize=(10,7))
sns.barplot(y='emotion_preds', x='counts', data=char_emotions[char_emotions['character_name']=='Rubeus Hagrid'], palette='pink_d')
plt.title('Rubeus Hagrid Most Common Emotions', fontsize=20)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Emotion', fontsize=15)
char_emotions = pd.DataFrame(hp.groupby('character_name').emotion_preds.value_counts())
char_emotions = char_emotions.rename(columns={'emotion_preds': 'counts'})
char_emotions = char_emotions.reset_index()
plt.figure(figsize=(10,7))
sns.barplot(y='emotion_preds', x='counts', data=char_emotions[char_emotions['character_name']=='Albus Dumbledore'], palette='pink_d')
plt.title('Albus Dumbledore Most Common Emotions', fontsize=20)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Emotion', fontsize=15)