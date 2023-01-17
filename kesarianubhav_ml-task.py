# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import sys
sys.path.append('../input')
# print(os.listdir('news_dataset'))
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import word2vec
from sklearn.manifold import TSNE
from sklearn import metrics
import pandas as pd 
import seaborn as sns
import time
# from wordcloud import STOPWORDS,
df=pd.read_json('../input/news-dataset/News_Category_Dataset.json',lines=True)
print(df.head())
print(df.info())
df.any().isnull()
df['link'].head(5)
df['headline'].head(10)
def category_counter(column):
    y=df[column].value_counts()
    x=set(df[column].values)
#     sns.countplot(x,y)
    print(y)
category_counter('category')
sns.countplot(x="category", data=df)
from wordcloud import WordCloud, STOPWORDS
def wordcloud_generator(column):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
                              background_color='white',
                              stopwords=stopwords,
                              max_words=500,
                              max_font_size=50, 
                              random_state=42
                             ).generate(str(df[column]))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title("Title")


#WordCloud for the Authors
wordcloud_generator('authors')
#wordcloud for the headlines
wordcloud_generator('headline')
all_text_description = ' '.join([text for text in df['short_description']])
all_text_headline = ' '.join([text for text in df['headline']])
print('Number of words in description in all_text:', len(all_text_description))
print('Number of words in headline in all_text:', len(all_text_headline))
df['news']=df['headline']+' '+df['short_description']
def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
# df['headline'] = df.apply(lambda row: decontracted(row['headline']), axis=1)
# df['short_description']=df.apply(lambda row: decontracted(row['short_description']), axis=1)
df['news'] = df.apply(lambda row: decontracted(row['news']), axis=1)
# print(decontracted('I\'ll know that can\'t have it'))
df.head(10)
# df['news'][0]
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['category']=le.fit_transform(df['category'])
df['category']
all_text_in_headlines=''.join([p for p in df['headline']])
all_text_in_descriptions=''.join([p for p in df['short_description']])
import string
punctuations=string.punctuation
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

# df['headline'] = df['headline'].apply(lambda x: remove_punct(x))
# df['short_description']=df['short_description'].apply(lambda x:remove_punct(x))
df['news']=df['news'].apply(lambda x:remove_punct(x))
df.head(10)
import nltk
# df['tokenized_headline'] = df.apply(lambda row: nltk.word_tokenize(row['headline']), axis=1)
# df['tokenized_short_description']=df.apply(lambda row: nltk.word_tokenize(row['short_description']), axis=1)
df['tokenized_news']=df.apply(lambda row:nltk.word_tokenize(row['news']),axis=1 )
df.head(3)
#def getting_clean_url()
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
# df['tokenized_headline'] = df['tokenized_headline'].apply(lambda x: remove_stopwords(x))
# df['tokenized_short_description']=df['tokenized_short_description'].apply(lambda x: remove_stopwords(x))
df['tokenized_news']=df['tokenized_news'].apply(lambda x: remove_stopwords(x))
df['news']=df['news'].apply(lambda x:remove_stopwords(x))
ps = nltk.PorterStemmer()
def text_stemmer(text):
    text = [ps.stem(word) for word in text]
    return text
# df['tokenized_headline'] = df['tokenized_headline'].apply(lambda x: text_stemmer(x))
# df['tokenized_short_description']=df['tokenized_short_description'].apply(lambda x: text_stemmer(x))
df['tokenized_news']=df['tokenized_news'].apply(lambda x: text_stemmer(x))
df['news']=df['news'].apply(lambda x:text_stemmer(x))
df['tokenized_news'].head(5)
from nltk import WordNetLemmatizer,pos_tag
wn1=WordNetLemmatizer()
def penn2morphy(penntag):
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 
    
def text_lemmatizer(text):
    return [wn1.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(text)]
# df['tokenized_headline'] = df['tokenized_headline'].apply(lambda x: text_lemmatizer(x))
# df['tokenized_short_description']=df['tokenized_short_description'].apply(lambda x: text_lemmatizer(x))
df['tokenized_news']=df['tokenized_news'].apply(lambda x: text_lemmatizer(x))
# df['news']=df['news'].apply(lambda x:text_lemmatizer(x))
df.head(10)
from keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer()
tokenizer.fit_on_texts(df.tokenized_news)
X_data=tokenizer.texts_to_sequences(df.tokenized_news)
# print(X_data[0])
word_index = tokenizer.word_index
print(type(word_index))
print('Total %s unique tokens.' % len(word_index))
print(X_data[0])
print(df['tokenized_news'][0])
embed_size=50
input_length=100
embeddings_index = {}
f = open('../input/glove50/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Total %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
absent_words = 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        absent_words += 1
print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)), '% of total words')
print(embedding_matrix[122])
print(embeddings_index['there'])
max_features=200000
max_senten_len=75
max_senten_num=4
embed_size=50
VALIDATION_SPLIT = 0.1
news=df['tokenized_news']
print(news.head())
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index)+1,
                            embed_size,
                            input_length=max_senten_len,
                            trainable=False)

from keras.preprocessing import sequence
# df['embedding_indexed_news']=df['tokenized_news'].apply(lambda x: embedding_m)
X = list(sequence.pad_sequences(X_data, maxlen=max_senten_len))
from keras import utils
print(len(X))
X=np.array(X)
print(X.shape)
Y=utils.np_utils.to_categorical(df.category)
print(Y.shape)
print(Y[0])
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

X,Y = shuffle(X,Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
from keras.layers import Input , Dense , LSTM,GlobalAveragePooling1D,GlobalMaxPooling1D,Bidirectional,LSTM,Conv1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras.models import Sequential, Model, load_model
def classifier():
    inp = Input(shape=(max_senten_len,), dtype='int32')
    x = embedding_layer(inp)
    x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))(x)
    outp = Dense(31, activation="softmax")(x)
    BiLSTM = Model(inp, outp)
    BiLSTM.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
    print(BiLSTM.summary())
    return BiLSTM
    
model1=classifier()
print(X_train[1])
histo=model1.fit(X_train,Y_train,batch_size=128,epochs=2,validation_split=0.25)
