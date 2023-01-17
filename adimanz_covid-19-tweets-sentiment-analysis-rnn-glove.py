!cp -r ../input/tweetcovid19/* ./
#Import the necessary libraries
import pandas as pd
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
import warnings 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
df = pd.read_csv('data/covid_19_tweets.CSV')
df.head()
df.drop(['status_id', 'user_id', 'screen_name'], axis = 1, inplace = True)
num_tweets = len(df)
df.isnull().sum()/num_tweets * 100
missing_cols = list(df.columns[(df.isnull().sum()/num_tweets * 100) > 85.0])

df.drop(missing_cols, axis = 1, inplace = True)

from iso639 import languages

def get_language(x):
    try:
        return languages.get(alpha2=x).name 
    except KeyError:
        return x
df['language'] = df['lang'].apply(lambda x: get_language(x))
df['language'].value_counts()[:10]
df['language'] = df['language'].str.replace('und','Undefined')
plt.figure(figsize = (15,7))
sns.barplot(x = df['language'].value_counts()[:10].index , y = df['language'].value_counts()[:10]/num_tweets*100)
plt.xlabel('Language', fontsize = 20)
plt.ylabel('Percentage of Tweets', fontsize = 20)
plt.xticks(fontsize = 15)
plt.title('Top Ten Languages with most Tweets', fontsize=20)
plt.show()
df_eng = df[df['language'] == 'English']
df_eng['language'].value_counts()
df_eng['text'][1128]
from clean_text import CleanText 
clean = CleanText()
df_eng['text_clean'] = clean.clean(df_eng['text']) #clean() removes urls, emoticons and hashtags
df_eng['text_clean'][1128]
df_eng['text_clean'] = df_eng['text_clean'].apply(lambda x: clean.tokenize(x)) #remove punctuations, stopwords, lemmatize and splits the sentences into tokens
df_eng['text_clean'][1128]
#Saving the dataframe as a pickle file to resume where I left off incase the kernel crashes or if I have to continue some other day
df_eng.to_pickle('pickle_files/tweets_eng.pkl') #Also reading and writing pickle files are much faster than csv
df_eng = pd.read_pickle('pickle_files/tweets_eng.pkl')
docs = df_eng['text_clean']

#tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1

#encode the documents
encoded_docs = t.texts_to_sequences(docs)

#pad docs to max length
padded_docs = pad_sequences(encoded_docs, maxlen = 22, padding = 'post') 
# Loading the classifier 
classifier = keras.models.load_model('Models/sentiment_classifier4.h5') #Negative: 0, Neutral: 1, Postive: 2
labels_categorical = classifier.predict(padded_docs) # Predicting the Sentiments of the Covid-19 tweets
labels_categorical[:10] #Output of each class by the softmax function
np.argmax(labels_categorical[:10], axis = 1) #np.argmax to get labels of the classes Negative: 0, Neutral: 1, Postive: 2
df_eng['labels'] = np.argmax(labels_categorical, axis = 1)
df_eng.to_pickle('pickle_files/final_df.pkl') 
df_eng = pd.read_pickle('pickle_files/final_df.pkl')
def label_to_sentiment(label):
    if label == 0:
        return 'Negative'
    elif label == 1:
        return 'Neutral'
    else:
        return 'Positive'
df_eng['sentiment'] = df_eng['labels'].apply(lambda x: label_to_sentiment(x))
pd.set_option('max_colwidth', 200)
df_eng[['text','sentiment']].iloc[368:373] #Let's check some random tweets to see if the predicted sentiments make sense
plt.figure(figsize = (15,7))
sns.barplot(x = df_eng['sentiment'].value_counts().index, y = df_eng['sentiment'].value_counts()/len(df_eng)*100)
plt.xlabel('Sentiment', fontsize = 20)
plt.ylabel('Percentage of Tweets(%)', fontsize = 20)
plt.xticks(fontsize = 15)
plt.title('Distribution of Tweets based on Sentiment', fontsize = 20)
plt.show()
from wordcloud import WordCloud
def plot_wordcloud(data):
    words = []
    for sent in data:
        for word in sent:
            words.append(word) 
    words = pd.Series(words).str.cat(sep=' ')
    wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(words)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
plot_wordcloud(df_eng['text_clean'][df_eng['sentiment'] == 'Positive'])
plot_wordcloud(df_eng['text_clean'][df_eng['sentiment'] == 'Negative'])
import nltk
import re
def extract_hashtag(text):
    hashtags=[]
    for i in text:
        ht=re.findall(r'#(\w+)',i)
        hashtags.append(ht)
    return hashtags
all_hashtags=extract_hashtag(df_eng.text)
def df_hashtag(sentiment_label):
    hashtags=extract_hashtag(df_eng.text[df_eng['sentiment']==sentiment_label])
    ht_fredist=nltk.FreqDist(sum(hashtags,[]))
    df_ht=pd.DataFrame({'Hashtag':list(ht_fredist.keys()),'Count':list(ht_fredist.values())})
    return df_ht
#Hashtags dataframes
ht_neg_df=df_hashtag('Negative')
ht_neu_df=df_hashtag('Neutral')
ht_pos_df=df_hashtag('Positive')
ht_neg_df.to_pickle('ht_neg_df.pkl')
ht_neu_df.to_pickle('ht_neu_df.pkl')
ht_pos_df.to_pickle('ht_pos_df.pkl')
ht_neg_df = pd.read_pickle('pickle_files/ht_neg_df.pkl')
ht_neu_df = pd.read_pickle('pickle_files/ht_neu_df.pkl')
ht_pos_df = pd.read_pickle('pickle_files/ht_pos_df.pkl')
def plot_hashtag(df,title):
    data=df.nlargest(columns="Count",n=20)
    plt.figure(figsize=(16,5))
    ax=sns.barplot(data=data,x='Hashtag',y='Count')
    plt.suptitle(title, fontsize=20)
    plt.xlabel('Hashtag', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.xticks(rotation=90)
    plt.tick_params(labelsize=15)
    plt.show()
plot_hashtag(ht_pos_df,'Positive sentiments')
plot_hashtag(ht_neg_df,'Negative sentiments')
plot_hashtag(ht_neu_df,'Neutral sentiments')
plt.figure(figsize = (15,8))
df_eng.groupby(['sentiment'])['favourites_count'].mean().plot(color='red', linestyle='dashed', marker='o',
                                                            markerfacecolor='red', markersize=10)
plt.title('Sentiment-wise Likes ratio', fontsize = 20)
plt.xlabel('Sentiment', fontsize = 20)
plt.ylabel('Average Likes', fontsize = 20)
plt.xticks(fontsize = 15)
plt.show()
plt.figure(figsize = (15,8))
df_eng.groupby(['sentiment'])['retweet_count'].mean().plot(color='green', linestyle='dashed', marker='o',
                                                            markerfacecolor='g', markersize=10)
plt.title('Sentiment-wise Retweets ratio', fontsize = 20)
plt.xlabel('Sentiment', fontsize = 20)
plt.ylabel('Average Retweets', fontsize = 20)
plt.xticks(fontsize = 15)
plt.show()
df_eng.sort_values(by = 'favourites_count', ascending = False).iloc[:3][['text','favourites_count']]
df_eng.sort_values(by = 'retweet_count', ascending = False).iloc[:3][['text','retweet_count']]
df_eng['time'] = pd.to_datetime(df_eng['created_at'])
df_eng.groupby(['time'])['text'].count().plot(marker='.', alpha=0.5, figsize=(15, 5))
plt.xlabel('Time', fontsize = 20)
plt.ylabel('Tweet Count', fontsize = 20)
plt.xticks(fontsize = 12)
plt.title('Rate of Overall Tweets', fontsize = 20)
plt.show()
df_eng[df_eng['sentiment'] == 'Positive'].groupby(['time'])['text'].count().plot(marker='.', alpha=0.5, figsize=(15, 5),
                                                                                 color = 'g',markerfacecolor='g')
plt.xlabel('Time', fontsize = 20)
plt.ylabel('Tweet Count', fontsize = 20)
plt.xticks(fontsize = 12)
plt.title('Rate of Positive Tweets', fontsize = 20)
plt.show()
df_eng[df_eng['sentiment'] == 'Negative'].groupby(['time'])['text'].count().plot(marker='.', alpha=0.5, figsize=(15, 5),
                                                                                 color = 'r',markerfacecolor='r')
plt.xlabel('Time', fontsize = 20)
plt.ylabel('Tweet Count', fontsize = 20)
plt.xticks(fontsize = 12)
plt.title('Rate of Negative Tweets', fontsize = 20)
plt.show()
# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our corpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(docs)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel
# Running and Trainign LDA model on the document term matrix.
ldamodel2 = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=10, iterations=10) 
for idx, topic in ldamodel2.show_topics(formatted=False, num_words= 30):
    print('Topic: {} \nWords: {}'.format(idx+1, '|'.join([w[0] for w in topic])))