%matplotlib inline



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf

import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM

from keras.preprocessing import sequence

from keras import backend as K



import nltk

nltk.download('punkt')

nltk.download('stopwords')

from nltk.corpus import stopwords

from wordcloud import WordCloud

import string

import re

df = pd.read_csv("../input/amazon-fine-food-reviews/Reviews.csv")

df.shape[0]
df = df[:50000]
df.shape
df.head()
df.info()
len(df.ProductId.unique())
len(df.UserId.unique())
#Lets do the value count on 'Scores'.

df.Score.value_counts()
plt.figure()

sns.countplot(x='Score', data=df, palette='RdBu')

plt.xlabel('Score (Rating)')

plt.show()
#copying the original dataframe to 'temp_df'.

temp_df = df[['UserId','HelpfulnessNumerator','HelpfulnessDenominator', 'Summary', 'Text','Score']].copy()



#Adding new features to dataframe.

temp_df["Sentiment"] = temp_df["Score"].apply(lambda score: "positive" if score > 3 else \

                                              ("negative" if score < 3 else "not defined"))

temp_df["Usefulness"] = (temp_df["HelpfulnessNumerator"]/temp_df["HelpfulnessDenominator"]).apply\

(lambda n: ">75%" if n > 0.75 else ("<25%" if n < 0.25 else ("25-75%" if n >= 0.25 and\

                                                                        n <= 0.75 else "useless")))



temp_df.loc[temp_df.HelpfulnessDenominator == 0, 'Usefulness'] = ["useless"]

# Removing all rows where 'Score' is equal to 3

#temp_df = temp_df[temp_df.Score != 3]

#Lets now observe the shape of our new dataframe.

temp_df.shape
sns.countplot(x='Sentiment', order=["positive", "negative"], data=temp_df, palette='RdBu')

plt.xlabel('Sentiment')

plt.show()
pos = temp_df.loc[temp_df['Sentiment'] == 'positive']

pos = pos[0:25000]



neg = temp_df.loc[temp_df['Sentiment'] == 'negative']

neg = neg[0:25000]
def create_Word_Corpus(temp):

    words_corpus = ''

    for val in temp["Summary"]:

        text = str(val).lower()

        #text = text.translate(trantab)

        tokens = nltk.word_tokenize(text)

        tokens = [word for word in tokens if word not in stopwords.words('english')]

        for words in tokens:

            words_corpus = words_corpus + words + ' '

    return words_corpus

        

# Generate a word cloud image

pos_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(pos))

neg_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(neg))

# Plot cloud

def plot_cloud(wordCloud):

    plt.figure( figsize=(16,8), facecolor='w')

    plt.imshow(wordCloud)

    plt.axis("off")

    plt.tight_layout(pad=0)

    plt.show()
plot_cloud(pos_wordcloud)
plot_cloud(neg_wordcloud)
#Checking the value count for 'Usefulness'

temp_df.Usefulness.value_counts()
df.head()
user = df['UserId'] == "AR5J8UI46CURR"

score = df['Score'] != 3

df[user & score]
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

stop_words=stopwords.words('english')
from nltk.stem import SnowballStemmer

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

lemmatizer = nltk.stem.WordNetLemmatizer()

stemmer=SnowballStemmer("english")



df['CleanedText'] = df['Text'].replace(to_replace=r'@\S+',value="",regex=True)

df["CleanedText"] = df['CleanedText'].replace(to_replace=r'[^A-Za-z0-9]+',value=" ",regex=True)

df["CleanedText"] = df["CleanedText"].apply(lambda x: x.split())

df["CleanedText"] = df['CleanedText'].apply(lambda x: [item for item in x if item not in stop_words])

df['CleanedText'] = df['CleanedText'].apply(lambda x: [stemmer.stem(w) for w in x])

df['CleanedText'] = df['CleanedText'].apply(' '.join)
df.head()
df["sentiment"] = df["Score"].apply(lambda score: "positive" if score >= 3 else "negative")
df.head()
#Sorting data according to ProductId in ascending order

df = df.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
df
df = df.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

df.shape
df.head()
sentences = np.array(df['CleanedText'])



for i,sentence in enumerate(sentences):

    sentences[i] = sentence

    

print(sentences[0])

print(sentences.shape)
# Find all words in the cleanedtext

vocab = set()

for x in sentences:

    for word in x.split():

        vocab.add(word)

        

len(vocab)
# Create a dictionary from vocabulary.

vocab_dict = dict.fromkeys(vocab, 0)
# Calculate count of each word..

for x in sentences:

    for word in x.split():

        vocab_dict[word]+=1
k = []

v = []

for keys,val in vocab_dict.items():

    k.append(keys)

    v.append(val)
kv = np.vstack((k,v)).T
kv.shape
df2 = pd.DataFrame(columns=["Word","Count"], data=kv)

df2['Count'] = df2['Count'].astype('int')

df2.head()
# Sort the dataframe to get the largest count at first place

df2.sort_values(by=['Count'], ascending=False, inplace=True)

df2.head()
# Give numbering to the most frequent word as 1 then next as 2 and so on.

df2.reset_index(inplace=True)

df2['mapping'] = df2.index + 1

df2.head()
df2.drop(columns=['index','Count'], inplace=True)

df2.head()
# Convert to dictionary for easier processing.

dictionary = dict(zip(df2['Word'], df2['mapping']))

len(dictionary)
df2.head()
del(df2)

del(sentences)

del(k)

del(v)

del(kv)
def change(x):

    l = list()

    for words in x.split():

        l.append(dictionary[words])

        

    return l
df
# Get LSTM Feature....

df['lstm_feature'] = df['CleanedText'].apply(change)

df.head()
df = df[['lstm_feature', 'sentiment']]

df.head()
df.head()
df['sentiment'] = df.sentiment.replace(to_replace=['positive', 'negative'], value=[1,0])

df.head()
X,y = df['lstm_feature'], df['sentiment']
# Find maximum length vector in LSTM Feature

m = 0

for x in df['lstm_feature']:

    m = max(m, len(x))

print(m)
max_review_length = 1500

X = sequence.pad_sequences(np.array(X), maxlen=max_review_length)
df.head()
X[0].shape
X.shape
# create the model

embedding_vecor_length = 32

total_words = 74581

model = Sequential()

model.add(Embedding(total_words, embedding_vecor_length, input_length=1500))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
X[0]
X = tf.convert_to_tensor(X, dtype=tf.float32)

y = tf.convert_to_tensor(y, dtype=tf.float32)
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')

mc = keras.callbacks.ModelCheckpoint('.mdl_wts.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=2)
# X.reset_index()
df.head()
history = model.fit(X, y, nb_epoch=10, batch_size=1024, validation_split=0.2, callbacks=[es, mc])
# Loss Curves

plt.figure(figsize=[8,6])

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

 

# Accuracy Curves

plt.figure(figsize=[8,6])

plt.plot(history.history['accuracy'],'r',linewidth=3.0)

plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)

plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)
import re



def preprocess(text):

    text = re.sub(r'@\S+', "", text)

    text = re.sub(r'[^A-Za-z0-9]+', " ", text)

    text = text.split()

    text = [item for item in text if item not in stop_words]

    text = [stemmer.stem(w) for w in text]

    text = ' '.join(text)

    return text
def sample_predict(text):

    text = preprocess(text)

    text = change(text)

    return text
# def get_sentiment(arr):

#     sentiment = []

#     for sent in arr:

#         if sent>= 0.5:

#              sentiment.append("Positive")

#         else:

#             sentiment.append("Negative")

#     return sentiment 
text1 = "Good Seller, Console came with everything as specified on description. I'll upload some pictures so you can see the box and hardware quality. "

encoded_text1 = sample_predict(text1)



text2 = "Very poor product"

encoded_text2 = sample_predict(text2)



encoded_text = sequence.pad_sequences([encoded_text1, encoded_text2], maxlen=max_review_length)



predictions = model.predict(encoded_text)

# sentiment = get_sentiment(predictions)

print(predictions)
model.save("model.h5")
text1 = " is amazingly simple to use. What great fun! "

encoded_text1 = sample_predict(text1)



text2 = "Very poor product"

encoded_text2 = sample_predict(text2)



encoded_text = sequence.pad_sequences([encoded_text1, encoded_text2], maxlen=max_review_length)



predictions = model.predict(encoded_text)

# sentiment = get_sentiment(predictions)

print(predictions)