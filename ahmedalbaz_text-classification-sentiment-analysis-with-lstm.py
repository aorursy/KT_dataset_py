import numpy as np
import pandas as pd
import os
import nltk
import spacy
from wordcloud import WordCloud
import tensorflow as tf
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.metrics import AUC
from sklearn.metrics import confusion_matrix, classification_report
#loading training and testing dataframes
train_data = pd.read_csv('/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv', encoding='latin-1')
test_data = pd.read_csv('/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv', encoding='latin-1')
#preview of training dataframe
train_data.head(5)
#preview of testing dataframe
test_data.head(5)
#Non-Null Count and dtype of training dataframe
train_data.info()
#Non-Null Count and dtype of testing dataframe
test_data.info()
#descriptive statistics for training dataframe
train_data.describe()
#descriptive statistics for testing dataframe
test_data.describe()
def preprocess_dataframe(dataframe, name=None):
    
    """
    Function to preprocess dataframe: removes redundant columns, converts dates to datetime type, creates new columns for mentions and hashtags
    
    Parameters
    ----------
    dataframe: Pandas Dataframe
        a dataframe to preprocess
    name: str, default=None
        The name to assign to a dataframe
    
    Returns
    -------
    dataframe: Pandas Dataframe
        a preprocessed dataframe
    """
    
    dataframe = dataframe.drop(columns=['UserName', 'ScreenName', 'Location'])
    dataframe['TweetAt'] = pd.to_datetime(dataframe['TweetAt'])
    dataframe['mentions'] = pd.Series([[word for word in tweet.split() if word.startswith('@')] for tweet in dataframe['OriginalTweet'].values])
    dataframe['hashtags'] = pd.Series([[word for word in tweet.split() if word.startswith('#')] for tweet in dataframe['OriginalTweet'].values])
    
    if name!=None:
        dataframe.name = name
    
    return dataframe
#preprocessing training and testing dataframes
train_df = preprocess_dataframe(train_data, name='train')
test_df = preprocess_dataframe(test_data, name='test')
def sentiment_countplot(data, title, figsize=(8, 5)):
    
    """
    Function that creates countplots for sentiments in a dataframe
    
    Parameters
    ----------
    data: Pandas dataframe
        a dataframe for which to visualize sentiments
    title: str
        Title of the the plot
    figsize: tuple, default=(8, 5)
        The size of the figure
    """
    fig = plt.figure(figsize=(8, 5))
    sns.set_palette("RdYlGn")
    ax = sns.countplot(data=data,
                  x='Sentiment',
                  order=['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'])
    ax.set_title(title)
    total = data.shape[0]
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{}%'.format(int(np.round(height/total*100))),
                ha="center") 
#Distribution of sentiments in training dataframe
sentiment_countplot(train_df, 'Count of Sentiments in Train Data')
#Distribution of sentiments in testing dataframe
sentiment_countplot(test_df, 'Count of Sentiments in Test Data')
#Distribution of tweet counts by sentiment over time in training dataframe
train_df.groupby(['TweetAt', 'Sentiment'])['OriginalTweet'].count().unstack().plot(kind='area', figsize=(10, 5))
plt.title('Count of Tweets in 2020')
plt.ylabel('Tweet Count')
#Distribution of tweet counts by sentiment over time in testing dataframe
test_df.groupby(['TweetAt', 'Sentiment'])['OriginalTweet'].count().unstack().plot(kind='area', figsize=(10, 5))
plt.title('Count of Tweets in 2020')
plt.ylabel('Tweet Count')
REPLACE_BY_SPACE = re.compile('[/(){}\[\]\|,;&-_]') #punctuation to replace
def preprocess_text(text):
    
    """
    Function to preprocess text: removes links, punctuation, spaces, non-alpha words and stop_words
    
    Parameters
    ----------
    text: str
        a string to be preprocessed
        
    Returns
    -------
    text: str
        a preprocessed string
    """
    text = text.lower()                                    #lowercase
    text = re.sub(r"http\S+", "", text)                    #replace links with ""
    text = re.sub(r"\@\S+", "", text)                      #replace mentions with ""
    text = re.sub(r"#\S+", "", text)                       #replace hashtags with ""
    text = re.sub(r"won\'t", "would not", text)            #deal with contractions
    text = re.sub(r"n\'t", " not", text)                   #deal with contractions
    text = REPLACE_BY_SPACE.sub(' ', text)                 #replace punctuation with space
    text = [word.strip() for word in text.split()]         #strip space from words
    text = [word for word in text if len(word)>2]          #removing words less than 2 characters
    text = [word for word in text if word!='amp']          #removing twitter amp
    text = ' '.join(text)
    return text
#preprocessing text column in train and test dataframes
train_df['Tweet'] = train_df['OriginalTweet'].apply(preprocess_text)
test_df['Tweet'] = test_df['OriginalTweet'].apply(preprocess_text)
def generate_wordcloud(data, mode='Tweet', sentiments='all'):
    
    """
    
    Function that generates a wordcloud for a givens sentiment from a dataframe containing a text column
    
    Parameters
    ----------
    data: Pandas DataFrame
        a pandas dataframe with a text column
    mode: str, default='Tweet'
        name of column in dataframe
    sentiments: str, default='all'
        The sentiment type for which to generate a wordcloud.
        Must be one of ['all', 'positive', 'negative']
    filter_common: boolean, default=False
        Removes 
    """
    
    
    df = data.copy()
    
    if sentiments=='positive':
        df = df[df.Sentiment.isin(['Positive', 'Extremely Positive'])]
    if sentiments=='negative':
        df = df[df.Sentiment.isin(['Negative', 'Extremely Negative'])]
    
     
#     if mode=='OriginalTweet':
#         text = ' '.join([i for i in text if not i.lower().startswith('#') and not i.lower().startswith('@') and not i.lower().startswith('https')])
    if mode=='Tweet':
        text = df[mode].str.split(' ').values
        text = ' '.join([' '.join(i) for i in text])
        text = text.strip()
    else:
        text = df[mode].values
        text = ' '.join([' '.join(i) for i in text])
        text = text.strip()

    
    cloud = WordCloud().generate(text)
    plt.figure()
    plt.imshow(cloud)
    try:
        plt.title(data.name)
    except:
        pass
#plotting wordcloud for tweets of positive sentiments in training and testing dataframes
for df in [train_df, test_df]:
    generate_wordcloud(df, mode='Tweet', sentiments='positive')
#plotting wordcloud for tweets of negative sentiments in training and testing dataframes
for df in [train_df, test_df]:
    generate_wordcloud(df, mode='Tweet', sentiments='negative')
#plotting wordcloud for tweets of all sentiments in training and testing dataframes
for df in [train_df, test_df]:
    generate_wordcloud(df, mode='Tweet', sentiments='all')
#plotting wordcloud of mentions of positive sentiments in training and testing dataframes
for df in [train_df, test_df]:
    generate_wordcloud(df, mode='mentions', sentiments='positive')
#plotting wordcloud of mentions of negative sentiments in training and testing dataframes
for df in [train_df, test_df]:
    generate_wordcloud(df, mode='mentions', sentiments='negative')
#plotting wordcloud of mentions of all sentiments in training and testing dataframes
for df in [train_df, test_df]:
    generate_wordcloud(df, mode='mentions', sentiments='all')
def get_top_grams(dataframe, sentiment, n_grams=2, top=10):
    
    """
    Function that generates the top n_grams from a text column of dataframe that correspond to
    a particular sentiment
    
    Parameters
    ----------
    dataframe: Pandas dataframe
        dataframe with a text column
    sentiments: str
        The sentiment type for which to generate the top n_grams
        Must be one of ['all', 'negative', 'positive']
    n_grams: int, default=2
        The number of grams to generate
    top: int, default=10
        The number of most common words to display
    """
    
    sentiments = ['Positive', 'Extremely Positive', 'Neutral', 'Negative', 'Extremely Negative']
    
    if sentiments!='all':
        if sentiment=='positive':
            sentiments = ['Positive', 'Extremely Positive']
        if sentiment=='negative':
            sentiments = ['Negative', 'Extremely Negative']

    df = dataframe[dataframe['Sentiment'].isin(sentiments)]['Tweet'].str.split()
    
    text = [word for words_list in df.values for word in words_list]
    
    grams = nltk.ngrams(text, n=n_grams)
    
    dist = nltk.FreqDist(grams)
    
    print(dist.most_common(top))
#displaying top biggrams for positive tweets in training dataframe
get_top_grams(train_df, 'positive')
#displaying top bigrams for negative tweets in training dataframe
get_top_grams(train_df, 'negative')
#displaying top bigrams of positive tweets in testing dataframe
get_top_grams(test_df, 'positive')
#display top bigrams of negative tweets in testing dataframe
get_top_grams(test_df, 'negative')
#calculating number of unique words
unique_words = set([word for word_list in train_df['Tweet'].str.split().values for word in word_list])
num_unique_words = len(unique_words)
print(num_unique_words)
MAX_NB_WORDS = 20000 #maximum number of words to take from corpus
Tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS, oov_token='<oov>') #initializing tokenizer
Tokenizer.fit_on_texts(train_df['Tweet'].values) #fitting tokenizer on training_datase
word_to_ind = Tokenizer.word_index #extracting word to index mapping from tokenzier
#displaying word to index mapping
word_to_ind
# getting text sequences from training and testing dataframes
X_train = Tokenizer.texts_to_sequences(train_df['Tweet'].values)
X_test = Tokenizer.texts_to_sequences(test_df['Tweet'].values)
# calculating maximum length of sequences among both training and testing dataframes
MAXLEN = max([len(x) for x in X_train] + [len(x) for x in X_test])
#adding padding of zeros to obtain uniform length for all sequences
X_train_padded = sequence.pad_sequences(X_train, maxlen=MAXLEN)
X_test_padded = sequence.pad_sequences(X_test, maxlen=MAXLEN)
#encoding sentiment labels
Y_train = train_df['Sentiment'].values
Y_test = test_df['Sentiment'].values
encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.transform(Y_test)
#one-hot-encoding sentiment labels
Y_train_enc = to_categorical(Y_train)
Y_test_enc = to_categorical(Y_test)
print(MAXLEN)
print(MAX_NB_WORDS)
print(Y_train_enc.shape)
# defining embedding dimension
EMBEDDING_DIM = 32
LSTM_NODES = 128
#building sequential neural network
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAXLEN, mask_zero=True))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(LSTM_NODES, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(Y_train_enc.shape[1], activation='softmax'))
#displaying model architecture
model.summary()
#defining pr-auc metric
auc = AUC(curve='PR')
#compiling model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', auc])
#training model
history = model.fit(X_train_padded, Y_train_enc, validation_data=(X_test_padded, Y_test_enc), epochs=5, batch_size=256, use_multiprocessing=True, shuffle=True)
#evaluating model on test set
Y_pred = model.predict(X_test_padded)
Y_pred = np.argmax(Y_pred, axis=1)
#extract labels from encoder
labels = list(encoder.classes_)
#calculate and plot confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, fmt='g')
#printing classification report
print(classification_report(Y_test, Y_pred, target_names=labels))
#mapping 5 classes to 3 more specific classes
mapping = {
    "Extremely Positive": "Positive",
    "Extremely Negative": "Negative",
    "Positive": "Positive",
    "Neutral": "Neutral",
    "Negative": "Negative"
}

#encoding sentiment labels

Y_train = train_df['Sentiment'].values
Y_test = test_df['Sentiment'].values

Y_train = list(map(mapping.get, Y_train))
Y_test = list(map(mapping.get, Y_test))

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.transform(Y_test)

Y_train_enc = to_categorical(Y_train)
Y_test_enc = to_categorical(Y_test)
#building sequential neural network
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAXLEN, mask_zero=True))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(LSTM_NODES, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(Y_train_enc.shape[1], activation='softmax'))
#compiling model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', auc])
#fitting model
history = model.fit(X_train_padded, Y_train_enc, validation_data=(X_test_padded, Y_test_enc), epochs=5, batch_size=256, use_multiprocessing=True, shuffle=True)
#evaluating model on test set
Y_pred = model.predict(X_test_padded)
Y_pred = np.argmax(Y_pred, axis=1)
#extract labels from encoder
labels = list(encoder.classes_)
#calculate and plot confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, fmt='g')
#printing classification report
print(classification_report(Y_test, Y_pred, target_names=labels))