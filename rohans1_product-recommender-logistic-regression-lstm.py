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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

%pylab inline

%config InlineBackend.figure_formats = ['retina'] #include it if you have high denisty retina display

import seaborn as sns #as it gives 2x plots with matplotlib and ipython notebook

import plotly.offline as py #to drew plotly

color = sns.color_palette()#graphs from a 

import plotly.offline as py#command line

py.init_notebook_mode(connected=True) #to create offine grapgs with notebook

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

import os

from sklearn.metrics import confusion_matrix, classification_report



# change dir according to your dataset location

dir = '/kaggle/input/grammar-and-online-product-reviews/GrammarandProductReviews.csv'

df = pd.read_csv(dir)

df.head()
#data overivew

print('rows: ', df.shape[0])

print('columns: ', df.shape[1])

print('\nfeatures: ', df.columns.to_list())

print('\nmissing vlues: ', df.isnull().values.sum())

print('\nUnique values: \n', df.nunique())
#see the data types of different columns

df.info()
#see the sum of missing values in each columns

df.isnull().sum()
#drop the rows having null values for reviews text

df = df.dropna(subset=['reviews.text'])
#there are many duplicate reveiws (exact same comments in review.text)

#but I am not going to clean the data yet,so i just use the data as it is, to go through t process

df['reviews.text'].value_counts()[10:50]
#plot ratings frequency

plt.figure(figsize=[10,5]) #[width, height]

x = list(df['reviews.rating'].value_counts().index)

y = list(df['reviews.rating'].value_counts())

plt.barh(x, y)



ticks_x = np.linspace(0, 50000, 6) # (start, end, no of ticks)

plt.xticks(ticks_x, fontsize=10, family='fantasy', color='black')

plt.yticks(size=15)



plt.title('Distribution of ratings', fontsize=20, weight='bold', color='navy', loc='center')

plt.xlabel('Count', fontsize=15, weight='bold', color='navy')

plt.ylabel('Ratings', fontsize=15, weight='bold', color='navy')

plt.legend(['reviews Rating'], shadow=True, loc=4)

#Loc =1 topright, loc=2 topleft, loc=3 bottomleft, loc=4 bottom right, loc=9 topmiddle

#plt.grid() #add grid lines
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title=None):

    wordcloud = WordCloud(

        background_color = 'white',

        stopwords = stopwords,

        max_words=300,

        max_font_size=40,

        scale=3,

        random_state=1 ).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title:

        fig.subtitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)

    

    plt.imshow(wordcloud)

    plt.show()

    

show_wordcloud(df['reviews.text'])    
#alternate code, seems to u=yeild diffent results

wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=300, max_font_size=40,

                     scale=3, random_state=1).generate(str(df['reviews.text'].value_counts()))

plt.figure(figsize=(15,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
df['reviews.title'].value_counts()
show_wordcloud(df['reviews.title'])

#alternate code, semms to yield different results

wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=400, max_font_size=

                     40, scale=30, random_state=1).generate_from_frequencies((df['reviews.title'].value_counts()))

plt.figure(figsize=(15,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
# try to tokenize to individual word (uni-gram) - reviews.title

split_title = []

listCounts = []

split_title = [x.split(" ") for x in df['reviews.title'].astype(str)]

big_list = []

for x in split_title:

    big_list.extend(x)



listCounts = pd.Series(big_list).value_counts()



wordcloud = WordCloud(background_color='white', max_words=400, max_font_size=40, scale=30,

        random_state=1).generate((listCounts[listCounts > 2]).to_string())

plt.figure(figsize=(15, 15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
len(big_list) #reveiws.title
# try to tokenize to individual word (uni-gram) - reviews.text

split_title = []

listCounts = []

split_title = [x.split(" ") for x in df['reviews.text'].astype(str)]

big_list = []

for x in split_title:

    big_list.extend(x)



listCounts = pd.Series(big_list).value_counts()



wordcloud = WordCloud(background_color='white', max_words=400, max_font_size=40, scale=30,

        random_state=1).generate((listCounts[listCounts > 2]).to_string())

plt.figure(figsize=(15, 15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
len(big_list) #reviews.text
#let's see what are the popular categories, looks quite messy

df['categories'].value_counts()
#Let's see which are the popular products review

df['name'].value_counts()
#on the reviews.didpurchase column, replace 38,886 null filds with "Null"

df['reviews.didPurchase'].fillna('Null', inplace=True)
plt.figure(figsize=(10,8))

ax = sns.countplot(df['reviews.didPurchase'])

ax.set_xlabel(xlabel="Shoppers did purchase the product", fontsize=17)

ax.set_ylabel(ylabel='Count of Reviews', fontsize=17)

ax.axes.set_title('Number of Genuine Reviews', fontsize=17)

ax.tick_params(labelsize=13)

df['reviews.didPurchase'].value_counts()
#shoppers who did purchase the product and provided the reveiw = 5%

3681/70008
#not much info int the correlation map

sns.set(font_scale=1.4)

plt.figure(figsize=(10,5))

sns.heatmap(df.corr(), cmap='coolwarm', annot=True, linewidths=.5)
df1 = df[df['reviews.didPurchase'] == True]

df1['name'].value_counts()
df1['name'].value_counts()[0:10].plot('barh', figsize=[10,6], fontsize=20).invert_yaxis()
# filter most purchased product with 5 star rating

df1 = df1[df1['name'] == 'The Foodsaver174 10 Cup Fresh Container - Fac10-000']

df1 = df1[df1['reviews.rating']==5]

# keep relevant columns only

df1 = df1[[ 'reviews.rating', 'reviews.text']]

df1
from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer



all_text = df['reviews.text']

y = df['reviews.rating']
word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 3) )  # try 1,3

#     max_features=10000)

word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(all_text)
char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(2, 6),

    max_features=50000)

char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(all_text)



train_features = hstack([train_char_features, train_word_features])
import time 

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

seed = 71



X_train, X_test, y_train, y_test = train_test_split(train_features, y, test_size=0.3, random_state=seed)

print('X_train', X_train.shape)

print('y_train', y_train.shape)

print('X_test', X_test.shape)

print('y_test', y_test.shape)

from sklearn.ensemble import RandomForestClassifier

time1 = time.time()

classifier = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=seed, n_jobs=-1)

classifier.fit(X_train, y_train)

preds1 = classifier.predict(X_test)



time_taken = time.time() -time1

print('Time taken: {:.2f} seconds'.format(time_taken))
print("Random Forest Model accuracy", accuracy_score(preds1, y_test))

print(classification_report(preds1, y_test))

print(confusion_matrix(preds1, y_test))
# n_estimators=None



# Random Forest Model accuracy 0.7014504999295874

#               precision    recall  f1-score   support



#            1       0.52      0.74      0.61       784

#            2       0.16      0.79      0.27       120

#            3       0.16      0.51      0.24       419

#            4       0.25      0.46      0.33      2412

#            5       0.93      0.74      0.82     17568



#    micro avg       0.70      0.70      0.70     21303

#    macro avg       0.41      0.65      0.45     21303

# weighted avg       0.82      0.70      0.74     21303



# [[  578    74    37    40    55]

#  [   14    95     4     1     6]

#  [   23    36   212    94    54]

#  [   73    84   316  1114   825]

#  [  426   288   772  3138 12944]]
# n_estimators=300 

# Time Taken:  955

# Random Forest Model accuracy 0.7151105478101676

#               precision    recall  f1-score   support



#            1       0.41      0.90      0.56       510

#            2       0.18      1.00      0.31        99

#            3       0.11      0.95      0.19       150

#            4       0.14      0.74      0.24       826

#            5       0.99      0.71      0.83     19718



#    micro avg       0.72      0.72      0.72     21303

#    macro avg       0.37      0.86      0.42     21303

# weighted avg       0.94      0.72      0.79     21303



# [[  460    31    12     4     3]

#  [    0    99     0     0     0]

#  [    1     1   142     5     1]

#  [    8    16   122   613    67]

#  [  663   403  1037  3695 13920]]
import xgboost as xgb

time1 = time.time()



xgb = xgb.XGBClassifier(n_jobs=1)

xgb.fit(X_train, y_train)

preds2 = xgb.predict(X_test)



time_taken = time.time() - time1

print('Time taken: {:.2f} seconds'.format(time_taken))
# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,

#        max_depth=3, min_child_weight=1, missing=None, n_estimators=100,

#        n_jobs=1, nthread=None, objective='multi:softprob', random_state=0,

#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

#        silent=True, subsample=1)



# time taken 2410
# manual method to check accuracy, see first 100 predictions, around 70% correct prediction

for i in range(100):

    if preds2[i] == np.array(y_test)[i]:

        print('1', end=', ')   # correct prediction

    else:

        print('0', end=', ')   # wrong prediction
# manual method to check accuracy, see some prediction of rating

preds2[0:100: 5]
# manual method to check accuracy, see correct test label

np.array(y_test)[0:100: 5]
#manuel method to check accuray, check on all 21303 test data set

correct = 0

wrong = 0

for i in range(21303):

    if preds2[i] == np.array(y_test)[i]:

        correct += 1

    else:

        wrong += 1

print(correct+wrong)

print(correct/21303)
print("XGBoost Model accuracy", accuracy_score(preds2, np.array(y_test)))
print("XGBoost Model accuracy", accuracy_score(preds2, y_test))

print(classification_report(preds2, y_test))

print(confusion_matrix(preds2, y_test))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, cross_val_score

time1 = time.time()

logit = LogisticRegression(C=1, multi_class = 'ovr')

logit.fit(X_train, y_train)

preds3 = logit.predict(X_test)



time_taken = time.time() - time1

print('Time Taken: {:.2f} seconds'.format(time_taken))
print("Logistic Regression accuracy", accuracy_score(preds3, y_test))

print(classification_report(preds3, y_test))

print(confusion_matrix(preds3, y_test))
df['sentiment'] = df['reviews.rating'] < 4

from sklearn.model_selection import train_test_split

train_text, test_text, train_y, test_y = train_test_split(df['reviews.text'],df['sentiment'], test_size=0.2)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

from keras.optimizers import Adam
MAX_NB_WORDS = 20000



# get the raw text data

texts_train = train_text.astype(str)

texts_test = test_text.astype(str)



# finally, vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False)

tokenizer.fit_on_texts(texts_train)

sequences = tokenizer.texts_to_sequences(texts_train)

sequences_test = tokenizer.texts_to_sequences(texts_test)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
MAX_SEQUENCE_LENGTH = 200

#pad sequences are used to bring all sentences to same size.

# pad sequences with 0s

x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', x_train.shape)

print('Shape of data test tensor:', x_test.shape)
model = Sequential()

model.add(Embedding(MAX_NB_WORDS, 128))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,input_shape=(1,)))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.fit(x_train, train_y,

          batch_size=128,

          epochs=10,

          validation_data=(x_test, test_y))