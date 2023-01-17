# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd 
import os

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Dense, Embedding, LSTM, Input,Dropout,Bidirectional,GRU,RNN
from sklearn.model_selection import train_test_split
import re
from bs4 import BeautifulSoup
train=pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/nlp_train.csv')
test=pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/_nlp_test.csv')
pd.set_option('display.max_colwidth',-1)
train.head(5)
train["offensive_language"].unique()
print( "Number of tweets in train data : ", len(train))
print( "Number of tweets in test data : ", len(test))
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Removing the @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Removing the URL links
    tweet = result = re.sub(r"http\S+", "", tweet)
    # Keeping only letters
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    tweet=re.sub(r'[^\w\s]','',tweet, re.UNICODE)
    # Removing additional whitespaces
    tweet = re.sub(r" +", ' ', tweet)
    #to lower case
    tweet=tweet.lower()
    #function to split text into word
    tokens = word_tokenize(tweet)
    #removing stop words
    tokens = [w for w in tokens if not w in stop_words]
    #stemming the words
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token, "v") for token in tokens]
    tokens = " ".join(tokens)
    return tokens

def pre_process(data):
    data_clean = [clean_tweet(tweet) for tweet in data.tweet]
    df=pd.DataFrame(data_clean,columns=["tweet"])
    df_clean=data.copy()
    df_clean["tweet"]=df["tweet"]
    return df_clean
train_clean=pre_process(train)
test_clean=pre_process(test)
train_clean.head()
num_words = 6000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
print(train_clean['tweet'][1])
tokenizer.fit_on_texts(train_clean['tweet'].values)
X = tokenizer.texts_to_sequences(train_clean['tweet'].values)
print(X[1])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

max_length_of_text = 50
X = pad_sequences(X, maxlen=max_length_of_text)

print(word_index)
print("Padded Sequences: ")
print(X)
print(X[1])
reviews = train_clean.tweet.str.cat(sep=' ')
#function to split text into word
tokens = word_tokenize(reviews)
frequency_dist = nltk.FreqDist(tokens)
sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50]
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
plt.figure(figsize = (10, 20), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
print(test_clean['tweet'][1])
X_final = tokenizer.texts_to_sequences(test_clean['tweet'].values)
print(X_final[1])
X_final = pad_sequences(X_final, maxlen=max_length_of_text)
print(X_final)
X_final.shape
y=train["offensive_language"]
train["offensive_language"].value_counts()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
earlystop = EarlyStopping(patience=5)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
checkpointer = ModelCheckpoint(filepath='weights1.h5', verbose=1, monitor='val_loss',save_weights_only=True,save_best_only=True)
callbacks = [earlystop, learning_rate_reduction,checkpointer]
from keras import backend as K

def sigmoid_func(x):
  ans=3*K.sigmoid(x)
  return ans
embeddings_index = {}
with open(os.path.join('/kaggle/input/glove-global-vectors-for-word-representation/glove.twitter.27B.100d.txt'),errors='ignore',encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print('Found %s word vectors in pretrained word vector model.' % len(embeddings_index))
print('Dimensions of the vector space : ', len(embeddings_index['the']))
EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_length_of_text,
                            trainable=False)
lstm_out=128
batch_size=32
inputs2 = Input((max_length_of_text, ))
x2 = embedding_layer(inputs2)
x2 = Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(x2)
x2 = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))(x2)
x2 = Dense(1,activation=sigmoid_func)(x2)
model2 = Model(inputs2, x2)
print(model2.summary())
model2.compile(loss ="mean_squared_error", optimizer='adam',metrics = ['accuracy'])
model2.fit(X_train, y_train,validation_data=(X_test,y_test), batch_size = batch_size, epochs = 20,callbacks=callbacks)
model2.load_weights("weights1.h5")
prediction=model2.predict(X_final)
df_final=pd.DataFrame(prediction,columns=["offensive_language"])
test["offensive_language"]=df_final["offensive_language"]
df = test
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)

