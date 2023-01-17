import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam
from keras.models import model_from_json
import os
import warnings 

warnings.filterwarnings("ignore")

def clean_ColText(data, col, stem=True):
    """Takes dataframe and column name and returns a dataframe with cleaned strings in the form of a list. Stemming is an option."""
    df = data.copy()
    table = str.maketrans('', '', string.punctuation)
    df[col] = df[col].map(lambda x: x.translate(table)) #remove punctuation
    df[col] = df[col].map(lambda x: x.lower()) #lowercase
    df[col] = df[col].apply(word_tokenize) #tokenize
    stop_words = set(stopwords.words('english'))
    df[col] = df[col].map(lambda x: [y for y in x if not y in stop_words]) #remove stop words
    df[col] = df[col].map(lambda x: [y for y in x if y not in ["’","’","”","“","‘","—"]]) #remove smart quotes and other non alphanums
    if stem:
        porter = PorterStemmer()
        df[col] = df[col].map(lambda x: [porter.stem(y) for y in x])
        return df
    return df

def plot_wordcloud(text, title=None, max = 1000, size=(10,5), title_size=16):
    """plots wordcloud"""
    wordcloud = WordCloud(max_words=max).generate(text)
    plt.figure(figsize=size)
    plt.title(title, size=title_size)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")


def mk_token(txt, char_lvl=True, low_cs=True, punct=True, stops=None, stem=None):
    """function with options for text cleaning and tokenizing"""
    if not punct:
        table = str.maketrans('','', string.punctuation)
        txt = txt.translate(table)
    if char_lvl:
        txt = list(txt)
    else:
        txt = word_tokenize(txt)
    if low_cs:
        txt = [w.lower() for w in txt]
    if not (stops is None):
        stop_words = set(stops)
        txt = [w for w in txt if not w in stop_words]
    if not (stem is None):
        stemmer = stem
        txt = [porter.stem(w) for w in txt]
    return txt

def code_vocab(txt, forw=True):
    """Remove duplicate tokens and enumerate."""
    vocab = list(set(txt))
    ch_int = dict((c,i) for i, c in enumerate(vocab))
    int_ch = dict((i,c) for i, c in enumerate(vocab))
    if forw:
        return ch_int, int_ch

def seq_text(txt, seq_length=1):
    X, y = ([], [])
    for i in range(0, len(txt) - seq_length, 1):
        seq_in = txt[i:i+seq_length]
        seq_out = txt[i+seq_length]
        X.append(seq_in)
        y.append(seq_out)
    n_rows = len(X)
    X = np.reshape(X, (n_rows, seq_length, 1))
    X = X/float(len(set(txt)))
    y = np.array(keras.utils.to_categorical(y))
    return X, y
 
#Define a function which creates a keras LSTM model with hidden layers, activation functions, and dropout rates
def simple_LSTM(input_shape, nodes_per=[60], hidden=0, out=2, act_out='softmax', drop=True, d_rate=0.1):
	"""Generate a keras neural network with arbitrary number of hidden layers, activation functions, dropout rates, etc"""
	model = Sequential()
	#adding first hidden layer with 60 nodes (first value in nodes_per list)
	model.add(LSTM(nodes_per[0],input_shape=input_shape, return_sequences=True))
	if drop:
		model.add(Dropout(d_rate))
	try:
		if hidden != 0:
			for i,j in zip(range(hidden), nodes_per[1:]):
				model.add(LSTM(j))
				if drop:
					model.add(Dropout(d_rate))
		model.add(Dense(out,activation=act_out))
		return(model)
	except:
		print('Error in generating hidden layers')
        
pd.read_csv('../input/Donald-Tweets!.csv').head(3)
sid = SentimentIntensityAnalyzer()
df = pd.read_csv('../input/Donald-Tweets!.csv')
df = df[df['Type'] == 'text'][['Date','Time','Tweet_Text']]
df = df[~df['Tweet_Text'].str.startswith('RT')]
df['Clean'] = df['Tweet_Text'].map(lambda x: re.sub(r'#\S+', '', str(x)))
df['Clean'] = df['Clean'].map(lambda x: re.sub(r'http\S+', '', str(x)))
df['Clean'] = df['Clean'].map(lambda x: re.sub(r'@\S+', '', str(x)))
df['Sent'] = df['Clean'].map(lambda x: sid.polarity_scores(x)['compound'])

plt.figure(figsize=(15,10))
plt.title('KDE of Tweet Sentiments')
plt.xlabel('Sentiment')
sns.set(color_codes=True)
sns.distplot(df['Sent'], hist=False, rug=True)
plt.show()
df.index = pd.DatetimeIndex(df['Date'])
df['Time'] = pd.to_datetime(df['Time'])
data = df[['Sent']].groupby(df.index.dayofweek).mean()
data2 = df[['Sent']].groupby(df['Time'].dt.hour).mean()

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.title('Average Tweet Sentiment by Day of Week')
plt.xlabel('Time')
plt.ylabel('Sentiment')
sns.set(color_codes=True)
plt.xticks(np.arange(len(data.index)), ('Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'))
plt.plot(data.index,data['Sent'])

plt.subplot(1, 2, 2)
plt.title('Average Tweet Sentiment by Hour of the Day')
plt.xlabel('Time')
plt.ylabel('Sentiment')
sns.set(color_codes=True)
plt.xticks(np.arange(len(data2.index)))
plt.plot(data2.index,data2['Sent'])

plt.show()
txt = clean_ColText(df, 'Clean')
txt = ' '.join(sum([x for x in txt['Clean']],[]))
plot_wordcloud(txt, title='DT Tweets Stemmed', size=(20,10), title_size=30)
neg = df[(df.Time > '11:00') & (df.Time < '14:00') & (df.Sent < .1)]
txt = clean_ColText(neg, 'Clean')
txt = ' '.join(sum([x for x in txt['Clean']],[]))
plot_wordcloud(txt, title='keywords found during times with low avg sentiment score', size=(20,10), title_size=30)