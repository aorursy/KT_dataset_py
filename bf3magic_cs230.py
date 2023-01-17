# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from keras.initializers import Constant
from keras.optimizers import Adam
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#coachella = pd.read_csv('../input/coachella2015/Coachella-2015-2-DFE.csv', engine="python")
#coachella.head(3)
#co_df = coachella[["coachella_sentiment", "text"]]
#co_df = co_df[co_df["text"].str.contains("fuck")]
#co_df.head(10)
#co_df.shape
#twitter = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/train.csv')
#twitter.head(3)
#twitter = twitter[twitter["tweet"].str.contains("fuck")]
#twitter.shape
DATASET_ENCODING = "ISO-8859-1"
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_PATH = "../input/sentiment140/training.1600000.processed.noemoticon.csv"
CURSE_WORDS = ["asshole", "bitch", "crap", "cunt", "damn", "fuck", "hell", "shit", "slut", "nigga", "prick"]
twitter = pd.read_csv(DATASET_PATH, encoding = DATASET_ENCODING, names = DATASET_COLUMNS)
twitter.head(3)

twitter = twitter[["target", "text"]]

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
twitter['text'] = twitter['text'].apply(lambda x : remove_URL(x))
twitter['text'] = twitter['text'].apply(lambda x : remove_html(x))
twitter['text'] = twitter['text'].apply(lambda x : remove_emoji(x))
twitter['text'] = twitter['text'].apply(lambda x : remove_punct(x))
curse_words = ' ' + ' | '.join(CURSE_WORDS) + ' '
twitter['text'] = twitter['text'].apply(lambda x : ' ' + x + ' ')
curse_tweets = twitter[twitter["text"].str.contains(curse_words, case=False)]
curse_tweets.shape

curse_tweets.head(3)
curse_tweets['target'] = curse_tweets['target'].apply(lambda x : 1 if x == 4 else x)
x = curse_tweets.target.value_counts()
sns.barplot(x.index, x)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=curse_tweets[curse_tweets['target']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='red')
ax1.set_title('positive tweets')
tweet_len=curse_tweets[curse_tweets['target']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='green')
ax2.set_title('negative tweets')
fig.suptitle('Words in a tweet with curse words')
plt.show()
def create_corpus(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus
corpus=create_corpus(curse_tweets)
embedding_dict={}
with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()
MAX_LEN=35
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec
model=Sequential()

embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
forward_layer = LSTM(64, return_sequences=True)
backward_layer = LSTM(64, activation='relu', return_sequences=True,
                       go_backwards=True)
model.add(Bidirectional(forward_layer, backward_layer=backward_layer))
model.add(Bidirectional(LSTM (64,dropout=0.2)))
#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


optimzer=Adam(learning_rate=1e-3)

model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model.summary()
X_train_val, X_test, y_train_val, y_test = train_test_split(tweet_pad, curse_tweets['target'].values, test_size = 0.15, random_state = 42, stratify = curse_tweets['target'])
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.15/0.85, random_state = 42, stratify = y_train_val)
print('Shape of Train',X_train.shape)
print("Shape of Validation ",X_val.shape)
print("Shape of Test ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=128,epochs=15,validation_data=(X_val,y_val),verbose=1)
y_pred = model.predict(X_test)
print(y_pred.shape)
y_pred=np.round(y_pred).astype(int).reshape(y_test.shape)
accuracy = sum([p == y for p, y in zip(y_pred, y_test)]) / len(y_pred) * 100
print(accuracy)
import tensorflow_hub as hub
import bert
from BertLibrary import BertFTModel
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE

train_val, test = train_test_split(curse_tweets, test_size = TEST_SIZE, random_state = 42)
train, val = train_test_split(train_val, test_size = VAL_SIZE/(TRAIN_SIZE+VAL_SIZE), random_state = 42)
print("TRAIN size: ", len(train))
print("TEST size: ", len(test))
print("VAL size: ", len(val))

!mkdir dataset
train.sample(frac=1.0).reset_index(drop=True).to_csv('dataset/train.tsv', sep='\t', index=None, header=None)
val.to_csv('dataset/dev.tsv', sep='\t', index=None, header=None)
test.to_csv('dataset/test.tsv', sep='\t', index=None, header=None)
! cd dataset && ls
bert_layer=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=True)
