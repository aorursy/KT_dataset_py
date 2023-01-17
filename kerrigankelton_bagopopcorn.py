import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
import matplotlib.pyplot as plt

%matplotlib inline

import nltk                            # Cleaning the data

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

import re

import os
import zipfile



files=['/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip',

       '/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip',

       '/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip']



for file in files :

    zip = zipfile.ZipFile(file,'r')

    zip.extractall()

    zip.close()
train=pd.read_csv('/kaggle/working/labeledTrainData.tsv', header = 0, delimiter="\t")

test=pd.read_csv('/kaggle/working/testData.tsv', header = 0, delimiter="\t")
train.head(10)
test.head(10)
print(train.shape)

print(test.shape)
train.isnull().sum()
test.isnull().sum()
train_length=train['review'].apply(len)

test_length=test['review'].apply(len)



import matplotlib.pyplot as plt

import seaborn as sns

fig=plt.figure(figsize=(15,4))

fig.add_subplot(1,2,1)

sns.distplot((train_length),color='red')



fig.add_subplot(1,2,2)

sns.distplot((test_length),color='blue')
train['word_n'] = train['review'].apply(lambda x : len(x.split(' ')))

test['word_n'] = test['review'].apply(lambda x : len(x.split(' ')))



fig=plt.figure(figsize=(15,4))

fig.add_subplot(1,2,1)

sns.distplot(train['word_n'],color='red')



fig.add_subplot(1,2,2)

sns.distplot(test['word_n'],color='blue')
from wordcloud import WordCloud

cloud=WordCloud(width=800, height=600).generate(" ".join(train['review'])) # join function can help merge all words into one string. " " means space can be a sep between words.

plt.figure(figsize=(15,10))

plt.imshow(cloud)

plt.axis('off')
#Import the stopwords (common words) to be removed from the corpus



import re

import json

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer



from nltk.stem.porter import PorterStemmer

corpus = []

s = set(stopwords.words('english'))

s.remove('not')

print("Stopwords length", len(s))
train['review']=train['review'].apply(lambda x: BeautifulSoup(x,"html5lib").get_text())

test['review']=test['review'].apply(lambda x: BeautifulSoup(x,"html5lib").get_text())
train['review']=train['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))

test['review']=test['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))
test['Sentiment'] = test['id'].map(lambda x: 1 if int(x.strip('"').split('_')[1]) >=5 else 0)

y_test = test_df['Sentiment']

y_test.head(10)
test.drop(['Sentiment'],axis = 1,inplace = True)

test.head()
train.sentiment.value_counts()  # balanced data...
def clean_review(raw_rev):

    review_text = BeautifulSoup(raw_rev,'lxml').get_text()          # remove HTML

    review_text = re.sub('[^a-zA-Z]'," ",review_text)               # includes only words

    review_words = review_text.lower().split()              # splits words and converts it to lowercase

    

    Stop_words = set(stopwords.words("english"))                        

    

    mean_words = [w for w in review_words if not w in Stop_words]    # removes  stopwords..

    review = ' '.join(mean_words)

    

    return review
train['clean_review'] = train['review'].apply(clean_review)

test['clean_review'] = test['review'].apply(clean_review)

test.drop(['review'],axis = 1,inplace = True)

test.rename(columns = {'clean_review':'review'},inplace = True)

train['length_review'] = train['clean_review'].apply(len)

train.head()
test.head()
from wordcloud import WordCloud

cloud=WordCloud(width=800, height=600).generate(" ".join(train['review'])) # join function can help merge all words into one string. " " means space can be a sep between words.

plt.figure(figsize=(15,10))

plt.imshow(cloud)

plt.axis('off')
from keras.preprocessing import sequence,text

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense,Dropout,Embedding,LSTM,SpatialDropout1D,Bidirectional

from keras.utils import to_categorical
train_x = train.iloc[:,3].values

target = train.sentiment.values



from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split( train_x, target , test_size = 0.2, random_state = 42)
print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)
# max length of the review



r_len=[]

for text in train['clean_review']:

    word=word_tokenize(text)

    l=len(word)

    r_len.append(l)

    

MAX_REVIEW_LEN=np.max(r_len)

MAX_REVIEW_LEN
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



max_features = 6000

max_words = 350

batch_size = 128

epochs = 6

num_classes=1



tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(x_train))

x_train = tokenizer.texts_to_sequences(x_train)

x_val = tokenizer.texts_to_sequences(x_val)
x_train = sequence.pad_sequences(x_train, maxlen=max_words)

x_val = sequence.pad_sequences(x_val, maxlen=max_words)

x_test = tokenizer.texts_to_sequences(test['review'])

x_test = sequence.pad_sequences(x_test, maxlen=max_words)

print(x_train.shape,x_val.shape,x_test.shape)
test.head()
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')

    

def get_embed_mat(EMBEDDING_FILE, max_features,embed_dim):

    # word vectors

    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf8'))

    print('Found %s word vectors.' % len(embeddings_index))



    # embedding matrix

    word_index = tokenizer.word_index

    num_words = min(max_features, len(word_index) + 1)

    all_embs = np.stack(embeddings_index.values()) #for random init

    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), 

                                        (num_words, embed_dim))

    for word, i in word_index.items():

        if i >= max_features:

            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

    max_features = embedding_matrix.shape[0]

    

    return embedding_matrix
EMBEDDING_FILE = '../input/glove6b/glove.6B.300d.txt'

embed_dim = 300 #word vector dim

embedding_matrix = get_embed_mat(EMBEDDING_FILE,max_features,embed_dim)

print(embedding_matrix.shape)
model = Sequential()

model.add(Embedding(max_features, embed_dim, input_length=x_train.shape[1],weights=[embedding_matrix],trainable=True))

model.add(SpatialDropout1D(0.25))

model.add(Bidirectional(LSTM(128,return_sequences=True)))

model.add(Bidirectional(LSTM(64,return_sequences=False)))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=epochs, batch_size=batch_size, verbose=1)
prediction = model.predict(x_test)

y_pred = (prediction > 0.5)
from sklearn.metrics import f1_score, confusion_matrix

print('F1-score: {0}'.format(f1_score(y_pred, y_test)))

print('Confusion matrix:')

confusion_matrix(y_pred, y_test)
t_loss = history.history['loss']

v_loss = history.history['val_loss']

epochs = range(1,len(t_loss)+1)

plt.plot(epochs,t_loss, 'bo', label='Training loss')

plt.plot(epochs,v_loss, 'r--', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
t_acc = history.history['accuracy']

v_acc = history.history['val_accuracy']

plt.plot(epochs,t_acc,'bo',label='Training acc')

plt.plot(epochs,v_acc,'r--', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
results = model.evaluate(x_val, y_val)

test_acc=np.round(results[1]*100,decimals=2)

#test_loss=np.round(results[2]*100,decimals=2)

print('Test accuracy is',test_acc,'%')

#print('Test loss is',test_loss,'%')
test.to_csv('result.csv',index=False)
#test_df['Sentiment'] = y_test

#test_df.drop(['review'],axis = 1,inplace = True)



#test_df.to_csv('Submission.csv',index = False)