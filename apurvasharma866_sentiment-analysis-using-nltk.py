import numpy as np

import pandas as pd

import nltk

import random

import os

from os import path

from PIL import Image



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS



# Set Plot Theme

sns.set_palette([

    "#30a2da",

    "#fc4f30",

    "#e5ae38",

    "#6d904f",

    "#8b8b8b",

])

# Alternate # plt.style.use('fivethirtyeight')



# Pre-Processing

import string

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

import re

from nltk.stem import PorterStemmer



# Modeling

import statsmodels.api as sm

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *

from nltk.util import ngrams

from collections import Counter

from gensim.models import word2vec



# Warnings

import warnings

warnings.filterwarnings('ignore')
# Read and Peak at Data

df = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv",index_col =[0])

df.head()
#Number of unique values

df.nunique()
df.describe().T.drop("count",axis=1)
# Continous Distributions

f, ax = plt.subplots(1,3,figsize=(16,8), sharey=False)

sns.distplot(df.Age, ax=ax[0])

ax[0].set_title("Age Distribution")

ax[0].set_ylabel("Density")

sns.distplot(df["Positive Feedback Count"], ax=ax[1])

ax[1].set_title("Positive Feedback Count Distribution")

sns.distplot(np.log10((df["Positive Feedback Count"][df["Positive Feedback Count"].notnull()]+1)), ax=ax[2])

ax[2].set_title("Positive Feedback Count Distribution\n[Log 10]")

ax[2].set_xlabel("Log Positive Feedback Count")

plt.tight_layout()

plt.show()
df.info()
df.drop(labels =['Clothing ID','Title'],axis = 1,inplace = True)#Dropping unwanted columns
df[df['Review Text'].isnull()].shape

data = df[~df['Review Text'].isnull()]  #Dropping columns which don't have any review

data.shape
import plotly.express as px



px.histogram(data, x = data['Rating'], color = data["Recommended IND"])

px.histogram(data, x = data['Class Name'])
px.scatter(data, x="Age", y="Positive Feedback Count", facet_row="Recommended IND", facet_col="Rating",trendline="ols",category_orders={"Rating": [1,2,3,4,5],'Recommended IND':[0,1]})
px.box(data, x="Age", y="Division Name", orientation="h",color = 'Recommended IND')

err1 = data['Review Text'].str.extractall("(&amp)")

err2 = data['Review Text'].str.extractall("(\xa0)")

print('with &amp',len(err1[~err1.isna()]))

print('with (\xa0)',len(err2[~err2.isna()]))



data['Review Text'] = data['Review Text'].str.replace('(&amp)','')

data['Review Text'] = data['Review Text'].str.replace('(\xa0)','')



err1 = data['Review Text'].str.extractall("(&amp)")

print('with &amp',len(err1[~err1.isna()]))

err2 = data['Review Text'].str.extractall("(\xa0)")

print('with (\xa0)',len(err2[~err2.isna()]))
!pip install TextBlob

from textblob import *



data['polarity'] = data['Review Text'].map(lambda text: TextBlob(text).sentiment.polarity)

data['polarity']
px.histogram(data, x = 'polarity',color="Rating", opacity = 0.5)
px.box(data, y="polarity", x="Department Name", orientation="v",color = 'Recommended IND')
data['review_len'] = data['Review Text'].astype(str).apply(len)

px.histogram(data, x = 'review_len' ,color = "Recommended IND")
data['token_count'] = data['Review Text'].apply(lambda x: len(str(x).split()))

px.histogram(data, x = 'token_count',color = "Recommended IND")
sam = data.loc[data.polarity == 1,['Review Text']].sample(3).values

for i in sam:

    print(i[0])
sam = data.loc[data.polarity == 0.5,['Review Text']].sample(3).values

for i in sam:

    print(i[0])
sam = data.loc[data.polarity < 0,['Review Text']].sample(3).values

for i in sam:

    print(i[0])
negative = (len(data.loc[data.polarity <0,['Review Text']].values)/len(data))*100

positive = (len(data.loc[data.polarity >0.5,['Review Text']].values)/len(data))*100

neutral  = len(data.loc[data.polarity >0 ,['Review Text']].values) - len(data.loc[data.polarity >0.5 ,['Review Text']].values)

neutral = neutral/len(data)*100



from matplotlib import pyplot as plt 

plt.figure(figsize =(10, 7)) 

plt.pie([positive,negative,neutral], labels = ['Positive','Negative','Neutral'] , colors = ["green" ,"red" ,"mediumslateblue"])
from sklearn.feature_extraction.text import CountVectorizer

def top_n_ngram(corpus,n = None,ngram = 1):

    vec = CountVectorizer(stop_words = 'english',ngram_range=(ngram,ngram)).fit(corpus)

    bag_of_words = vec.transform(corpus) #Have the count of  all the words for each review

    sum_words = bag_of_words.sum(axis =0) #Calculates the count of all the word in the whole review

    words_freq = [(word,sum_words[0,idx]) for word,idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq,key = lambda x:x[1],reverse = True)

    return words_freq[:n]
common_words = top_n_ngram(data['Review Text'], 10,1)

df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

plt.figure(figsize =(10,5))

df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(

kind='bar', title='Top 10 unigrams in review after removing stop words')
common_words = top_n_ngram(data['Review Text'], 20,2)

df = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])

plt.figure(figsize =(10,5))

df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).plot(

kind='bar', title='Top 20 bigrams in review after removing stop words')
blob= TextBlob(str(data['Review Text']))

pos = pd.DataFrame(blob.tags,columns =['word','pos'])

pos1 = pos.pos.value_counts()[:20]

plt.figure(figsize = (10,5))

pos1.plot(kind='bar',title ='Top 20 Part-of-speech taggings')
y = data['Recommended IND']

X = data.drop(columns = 'Recommended IND')
import seaborn as sns

sns.heatmap(X.corr(),annot =True, cmap = "icefire")
set1 =set()

cor = X.corr()

for i in cor.columns:

    for j in cor.columns:

        if cor[i][j]>0.8 and i!=j:

            set1.add(i)

print(set1)

X = X.drop(labels = ['token_count'],axis = 1)

print("correlation: ", X.corr())
class1 =[]

for i in X.polarity:

    if float(i)>=0.0:

        class1.append(1)

    elif float(i)<0.0:

        class1.append(0)

X['sentiment'] = class1



X.groupby(X['sentiment']).describe().T
print("Shape of X: " , X.shape)

print("Shape of y: " , y.shape)
import nltk

import re

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer





corpus =[]

X.index = np.arange(len(X))
len(X)
from tqdm import tqdm

for i in tqdm(range(len(X))):

    review = re.sub('[^a-zA-z]',' ',X['Review Text'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review =[ps.stem(i) for i in review if not i in set(stopwords.words('english'))]

    review =' '.join(review)

    corpus.append(review)

corpus[0:5]
len(corpus)
from sklearn.feature_extraction.text import CountVectorizer as CV

cv  = CV(max_features = 3000,ngram_range=(1,1))

X_cv = cv.fit_transform(corpus).toarray()

y = y.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

from sklearn import metrics

acc = accuracy_score(y_test, y_pred)

print("Accuracy of the classifier: ",acc)

print("Confusion matrix is :\n",metrics.confusion_matrix(y_test,y_pred))

print("Classification report: \n" ,metrics.classification_report(y_test,y_pred))
from sklearn.feature_extraction.text import TfidfVectorizer as TV

tv  = TV(ngram_range =(1,1),max_features = 3000)

X_tv = tv.fit_transform(corpus).toarray()

X_train, X_test, y_train, y_test = train_test_split(X_tv, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Accuracy of the classifier: ",acc)

print("Confusion matrix is :\n",metrics.confusion_matrix(y_test,y_pred))

print("Classification report: \n" ,metrics.classification_report(y_test,y_pred))
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = 3000)

tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)

padded = pad_sequences(sequences, padding='post')

word_index = tokenizer.word_index

count = 0

for i,j in word_index.items():

    if count == 11:

        break

    print(i,j)

    count = count+1
embedding_dim = 64

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(3000, embedding_dim),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(6, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.summary()
num_epochs = 10



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(padded,y,epochs= num_epochs,validation_split= 0.39)
loss = model.history.history

loss = pd.DataFrame(loss)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Basic ANN Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = range(1,11)

ax1.plot(epoch_list, loss['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, loss['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 11, 1))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, loss['loss'], label='Train Loss')

ax2.plot(epoch_list, loss['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 11, 1))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
sample_string = "I hate it"

sample = tokenizer.texts_to_sequences(sample_string)

padded_sample = pad_sequences(sample, padding='post')

print("Padded sample", padded_sample.T)

print("Probabilty of a person recommending :",model.predict(padded_sample.T)[0][0]*100,"%")
sample_string = "i love the fabric"

sample = tokenizer.texts_to_sequences(sample_string)

padded_sample = pad_sequences(sample, padding='post')

print("Padded sample", padded_sample.T)

print("Probabilty of a person recommending :",model.predict(padded_sample.T)[0][0]*100,"%")
y = data['Recommended IND'].tolist()

X = list(data["Review Text"])



# Separate out the sentences and labels into training and test sets

training_size = int(len(X) * 0.8)



training_sentences = X[0:training_size]

testing_sentences = X[training_size:]

training_labels = y[0:training_size]

testing_labels = y[training_size:]



# Make labels into numpy arrays for use with the network later

training_labels_final = np.array(training_labels)

testing_labels_final = np.array(testing_labels)
vocab_size = 1000

embedding_dim = 16

max_length = 100

trunc_type='post'

padding_type='post'

oov_tok = "<OOV>"





from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences,maxlen=max_length, padding=padding_type, 

                       truncating=trunc_type)



testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequences,maxlen=max_length, 

                               padding=padding_type, truncating=trunc_type)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



def decode_review(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])



print(decode_review(padded[1]))

print(training_sentences[1])
# Build a basic sentiment network

# Note the embedding layer is first, 

# and the output is only 1 node as it is either 0 or 1 (negative or positive)

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(6, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10

model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
loss = model.history.history

loss = pd.DataFrame(loss)

loss




f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Basic ANN Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = range(1,11)

ax1.plot(epoch_list, loss['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, loss['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 11, 1))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, loss['loss'], label='Train Loss')

ax2.plot(epoch_list, loss['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 11, 1))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")