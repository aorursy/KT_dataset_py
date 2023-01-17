# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from textblob import TextBlob

from collections import Counter

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 

from nltk.stem import SnowballStemmer

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from keras.preprocessing import sequence

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Embedding

from keras.layers import LSTM
#Import the train data

train = pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

train.head()
train.author.unique()
#count the author frequencies of training data

plt.hist(train.author)

plt.title("Frequency of Authors Occurence",fontsize=15)

plt.xticks(np.arange(3),(['Edgar Allen Poe', 'Mary Shelley', 'HP Lovecraft']))

x_coor = [0,1.8,1]

for idx, label in enumerate(train.author.value_counts().index):

    cnt = train.author.value_counts()[idx]

    plt.text(x_coor[idx], cnt, str(cnt), color='black', fontweight='bold')
train.author.shape
train_qs = pd.Series(train['text'].tolist()).astype(str)

dist_train = train_qs.apply(len)

plt.figure(figsize=(15, 10))

plt.hist(dist_train, bins=200, range=[0, 200], normed=True,alpha=.7)

plt.title('Normalized histogram of character count in text', fontsize=15)

plt.legend()

plt.xlabel('Number of characters', fontsize=15)

plt.ylabel('Probability', fontsize=15)
print('mean num',dist_train.mean(),'\n','std num', dist_train.std(),'\n',

      'min num',dist_train.min(),'\n','max num', dist_train.max())
dist_train = train_qs.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(15, 10))

plt.hist(dist_train, bins=50, range=[0, 50], normed=True,alpha=.7)

plt.title('Normalised histogram of word count in texts', fontsize=15)

plt.legend()

plt.xlabel('Number of words', fontsize=15)

plt.ylabel('Probability', fontsize=15)

print('mean num',dist_train.mean(),'\n','std num', dist_train.std(),'\n',

      'min num',dist_train.min(),'\n','max num', dist_train.max())
from wordcloud import WordCloud

from stop_words import get_stop_words



stop_words = get_stop_words('en')

stop_words = get_stop_words('english')

cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(train.text.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
#Store the text of each author in a Python list

eap = train[train.author=="EAP"]["text"].values

hpl = train[train.author=="HPL"]["text"].values

mws = train[train.author=="MWS"]["text"].values
cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(eap.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(hpl.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(mws.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
train['author_num']=train['author'].apply({'EAP':0,  'HPL':1,'MWS':2}.get)
train.head()
raw_text_train=train['text'].values

raw_text_test=test['text'].values

author_train=train['author_num'].values

num_labels = len(np.unique(author_train))
num_labels
 #text pre-processing

stop_words = set(stopwords.words('english'))

stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

stemmer = SnowballStemmer('english')

print ("pre-processing train docs...")

processed_train = []

for doc in raw_text_train:

    tokens = word_tokenize(doc)

    filtered = [word for word in tokens if word not in stop_words]

    stemmed = [stemmer.stem(word) for word in filtered]

    processed_train.append(stemmed)
print ("pre-processing test docs...")

processed_test = []

for doc in raw_text_test:

    tokens = word_tokenize(doc)

    filtered = [word for word in tokens if word not in stop_words]

    stemmed = [stemmer.stem(word) for word in filtered]

    processed_test.append(stemmed)
processed_docs_all = np.concatenate((processed_train, processed_test), axis=0)


from gensim import corpora

dictionary = corpora.Dictionary(processed_docs_all)

dictionary_size = len(dictionary.keys())

print ("dictionary size: ", dictionary_size )

    #dictionary.save('dictionary.dict')

    #corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print ("converting to token ids...")

word_id_train, word_id_len = [], []

for doc in processed_train:

    word_ids = [dictionary.token2id[word] for word in doc]

    word_id_train.append(word_ids)

    word_id_len.append(len(word_ids))
word_id_test, word_ids = [], []

for doc in processed_test:

    word_ids = [dictionary.token2id[word] for word in doc]

    word_id_test.append(word_ids)

    word_id_len.append(len(word_ids))
seq_len = np.round((np.mean(word_id_len) + 2*np.std(word_id_len))).astype(int)
#pad sequences

word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)

word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)

y_train_enc = np_utils.to_categorical(author_train,num_labels)
print ("fitting LSTM ...")

model = Sequential()

model.add(Embedding(dictionary_size, 128, dropout=0.2))

model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))

model.add(Dense(num_labels))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(word_id_train, y_train_enc, nb_epoch=1, batch_size=250, verbose=1)

test_pred = model.predict_classes(word_id_test)
test_pred = model.predict_proba(word_id_test)
prob=pd.DataFrame(test_pred,columns=['EAP','HPL','MWS'])
prob.shape
test.head()
submit1=pd.concat([test, prob], axis=1)

del submit1['text']
submit1.to_csv('./lstm_sentiment.csv', index=False, header=True)
#submit1.head()
pd.read_csv("../input/sample_submission.csv").head()
x_train = pd.DataFrame(word_id_train)

x_test = pd.DataFrame(word_id_test)

y_train=author_train



from sklearn.cross_validation import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

import xgboost as xgb



# Set our parameters for xgboost

params = {}

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

param['nthread'] = 4

param['eval_metric'] = 'auc'



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist)

d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)