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
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
import numpy as np 
import pandas as pd 
import os

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Dropout, concatenate, Bidirectional
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

np.random.seed(42)
data = pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/nlp_train.csv')
data_test = pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/_nlp_test.csv')
pd.set_option('display.max_colwidth',-1)
data.head()
test_tweet = data_test['tweet']
tweets = pd.concat([data['tweet'], test_tweet],axis=0)
SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}

def clean(text, stem_words=True):
    import re
    from string import punctuation
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords
    
    def pad_str(s):
        return ' '+s+' '
    
    if pd.isnull(text):
        return ''

    
    if type(text) != str or text=='':
        return ''

    # Clean the text
    text = re.sub("\'s", " ", text) 
    text = re.sub("@[\w]*", "", text)
    text = re.sub("[^a-zA-Z#]", " " , text)
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    
    # add padding to punctuations and special chars, we still need them later
    
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    

        
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text) # replace non-ascii word with special word
    
    # indian dollar
    
    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rt", " ", text, flags=re.IGNORECASE)
    
    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)

    
    
    
    text = re.sub('[0-9]+\.[0-9]+', " NUMBER ", text)
  
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()
       # Return a list of words
    return text
tweets = tweets.apply(clean)
tweets.head()
tweets.tail()
num_words = 25000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
# print(data['text'][0])
tokenizer.fit_on_texts(tweets.values)
X = tokenizer.texts_to_sequences(tweets.values)
print(X[0])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


import matplotlib.pyplot as plt
maxlen = []
for i in range(0,len(X)):
    maxlen.append(len(X[i]))
        
plt.plot(maxlen)
max_length_of_text = 35
X = pad_sequences(X, maxlen=max_length_of_text)

print(word_index)
print("Padded Sequences: ")
print(X)
print(X[0])
y = data['offensive_language']
train = X[:13000]
test = X[13000:]
X_train, X_test, y_train, y_test = train_test_split(train,y, test_size = 0.2, random_state = 42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
embed_dim = 64 #Change to observe effects
lstm_out = 128 #Change to observe effects
batch_size = 16
embeddings_index = {}
f = open('/kaggle/input/glovetwitter100d/glove.twitter.27B.100d.txt')
#sub6,7 100d; sub8(6),9(7) 200d, sub 10,11 100d 
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
length_of_text = 35
inputs2 = Input((length_of_text, ))
x1 = embedding_layer(inputs2)
# x2 = LSTM(256, dropout=0.25, recurrent_dropout=0.25)(x1)
x3 = LSTM(512, dropout=0.4, recurrent_dropout=0.4)(x1)
# x4 = concatenate([x2,x3])
x5 = Dense(1, activation='linear')(x3)
model3 = Model(inputs2, x5)
print(model3.summary())
from keras.optimizers import Adadelta, Adam
model3.compile(loss = 'mean_squared_error', optimizer='nadam')
model3.fit(X_train, y_train, batch_size = batch_size, epochs = 8, validation_data=(X_test, y_test))
y_pred = model3.predict(test)
y_final = []
for pred in y_pred:
  if pred < 0:
    y_final.append(0)
  elif pred > 3.0:
    y_final.append(3.0)
  else:
    y_final.append(1*pred[0])
submission_df = pd.read_csv('/kaggle/input/nnfl-lab-3-nlp/_nlp_test.csv')
submission_df['offensive_language'] = y_final
create_download_link(submission_df)
# model3.save('sub16.h5')
