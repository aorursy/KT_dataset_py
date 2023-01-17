import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import re

from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score
fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")
fake.head()
#Counting by Subjects 

for key,count in fake.subject.value_counts().iteritems():

    print(f"{key}:\t{count}")

    

#Getting Total Rows

print(f"Total Records:\t{fake.shape[0]}")
plt.figure(figsize=(8,5))

sns.countplot("subject", data=fake)

plt.show()
#Word Cloud

text = ''

for news in fake.text.values:

    text += f" {news}"

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = set(nltk.corpus.stopwords.words("english"))).generate(text)

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()

del text
real = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")

real.head()
#First Creating list of index that do not have publication part

unknown_publishers = []

for index,row in enumerate(real.text.values):

    try:

        record = row.split(" -", maxsplit=1)

        #if no text part is present, following will give error

        record[1]

        #if len of piblication part is greater than 260

        #following will give error, ensuring no text having "-" in between is counted

        assert(len(record[0]) < 260)

    except:

        unknown_publishers.append(index)
#Thus we have list of indices where publisher is not mentioned

#lets check

real.iloc[unknown_publishers].text

#true, they do not have text like "WASHINGTON (Reuters)"
real.iloc[8970]

#yep empty

#will remove this soon
#Seperating Publication info, from actual text

publisher = []

tmp_text = []

for index,row in enumerate(real.text.values):

    if index in unknown_publishers:

        #Add unknown of publisher not mentioned

        tmp_text.append(row)

        

        publisher.append("Unknown")

        continue

    record = row.split(" -", maxsplit=1)

    publisher.append(record[0])

    tmp_text.append(record[1])
#Replace existing text column with new text

#add seperate column for publication info

real["publisher"] = publisher

real["text"] = tmp_text



del publisher, tmp_text, record, unknown_publishers
real.head()
#checking for rows with empty text like row:8970

[index for index,text in enumerate(real.text.values) if str(text).strip() == '']

#seems only one :)
#dropping this record

real = real.drop(8970, axis=0)
# checking for the same in fake news

empty_fake_index = [index for index,text in enumerate(fake.text.values) if str(text).strip() == '']

print(f"No of empty rows: {len(empty_fake_index)}")

fake.iloc[empty_fake_index].tail()
#Looking at publication Information

# Checking if Some part of text has been included as publisher info... No such cases it seems :)



# for name,count in real.publisher.value_counts().iteritems():

#     print(f"Name: {name}\nCount: {count}\n")
#Getting Total Rows

print(f"Total Records:\t{real.shape[0]}")



#Counting by Subjects 

for key,count in real.subject.value_counts().iteritems():

  print(f"{key}:\t{count}")
sns.countplot(x="subject", data=real)

plt.show()
#WordCloud For Real News

text = ''

for news in real.text.values:

    text += f" {news}"

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = set(nltk.corpus.stopwords.words("english"))).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()

del text
# Adding class Information

real["class"] = 1

fake["class"] = 0
#Combining Title and Text

real["text"] = real["title"] + " " + real["text"]

fake["text"] = fake["title"] + " " + fake["text"]
# Subject is diffrent for real and fake thus dropping it

# Aldo dropping Date, title and Publication Info of real

real = real.drop(["subject", "date","title",  "publisher"], axis=1)

fake = fake.drop(["subject", "date", "title"], axis=1)
#Combining both into new dataframe

data = real.append(fake, ignore_index=True)

del real, fake
# Download following if not downloaded in local machine



# nltk.download('stopwords')

# nltk.download('punkt')
y = data["class"].values

#Converting X to format acceptable by gensim, removing annd punctuation stopwords in the process

X = []

stop_words = set(nltk.corpus.stopwords.words("english"))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

for par in data["text"].values:

    tmp = []

    sentences = nltk.sent_tokenize(par)

    for sent in sentences:

        sent = sent.lower()

        tokens = tokenizer.tokenize(sent)

        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]

        tmp.extend(filtered_words)

    X.append(tmp)



del data
import gensim
#Dimension of vectors we are generating

EMBEDDING_DIM = 100



#Creating Word Vectors by Word2Vec Method (takes time...)

w2v_model = gensim.models.Word2Vec(sentences=X, size=EMBEDDING_DIM, window=5, min_count=1)
#vocab size

len(w2v_model.wv.vocab)



#We have now represented each of 122248 words by a 100dim vector.
#see a sample vector for random word, lets say Corona 

w2v_model["corona"]
w2v_model.wv.most_similar("iran")
w2v_model.wv.most_similar("fbi")
w2v_model.wv.most_similar("facebook")
w2v_model.wv.most_similar("computer")
#Feeding US Presidents

w2v_model.wv.most_similar(positive=["trump","obama", "clinton"])

#First was Bush
# Tokenizing Text -> Repsesenting each word by a number

# Mapping of orginal word to number is preserved in word_index property of tokenizer



#Tokenized applies basic processing like changing it yo lower case, explicitely setting that as False

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X)



X = tokenizer.texts_to_sequences(X)
# lets check the first 10 words of first news

#every word has been represented with a number

X[0][:10]
#Lets check few word to numerical replesentation

#Mapping is preserved in dictionary -> word_index property of instance

word_index = tokenizer.word_index

for word, num in word_index.items():

    print(f"{word} -> {num}")

    if num == 10:

        break        
# For determining size of input...



# Making histogram for no of words in news shows that most news article are under 700 words.

# Lets keep each news small and truncate all news to 700 while tokenizing

plt.hist([len(x) for x in X], bins=500)

plt.show()



# Its heavily skewed. There are news with 5000 words? Lets truncate these outliers :) 
nos = np.array([len(x) for x in X])

len(nos[nos  < 700])

# Out of 48k news, 44k have less than 700 words
#Lets keep all news to 700, add padding to news with less than 700 words and truncating long ones

maxlen = 700 



#Making all news of size maxlen defined above

X = pad_sequences(X, maxlen=maxlen)
#all news has 700 words (in numerical form now). If they had less words, they have been padded with 0

# 0 is not associated to any word, as mapping of words started from 1

# 0 will also be used later, if unknows word is encountered in test set

len(X[0])
# Adding 1 because of reserved 0 index

# Embedding Layer creates one more vector for "UNKNOWN" words, or padded words (0s). This Vector is filled with zeros.

# Thus our vocab size inceeases by 1

vocab_size = len(tokenizer.word_index) + 1
# Function to create weight matrix from word2vec gensim model

def get_weight_matrix(model, vocab):

    # total vocabulary size plus 0 for unknown words

    vocab_size = len(vocab) + 1

    # define weight matrix dimensions with all 0

    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    # step vocab, store vectors using the Tokenizer's integer mapping

    for word, i in vocab.items():

        weight_matrix[i] = model[word]

    return weight_matrix
#Getting embedding vectors from word2vec and usings it as weights of non-trainable keras embedding layer

embedding_vectors = get_weight_matrix(w2v_model, word_index)
#Defining Neural Network

model = Sequential()

#Non-trainable embeddidng layer

model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=maxlen, trainable=False))

#LSTM 

model.add(LSTM(units=128))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])



del embedding_vectors
model.summary()
#Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y) 
model.fit(X_train, y_train, validation_split=0.3, epochs=6)
#Prediction is in probability of news being real, so converting into classes

# Class 0 (Fake) if predicted prob < 0.5, else class 1 (Real)

y_pred = (model.predict(X_test) >= 0.5).astype("int")
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
del model
#invoke garbage collector to free ram

import gc

gc.collect()
from gensim.models.keyedvectors import KeyedVectors
# Takes RAM 

word_vectors = KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)

EMBEDDING_DIM=300
# word_vectors.most_similar('usa')
# word_vectors.most_similar('fbi')
# word_vectors.most_similar('Republic')
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in word_index.items():

    try:

        embedding_vector = word_vectors[word]

        embedding_matrix[i] = embedding_vector

    except KeyError:

        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)



del word_vectors 
model = Sequential()

model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], input_length=maxlen, trainable=False))

model.add(Conv1D(activation='relu', filters=4, kernel_size=4))

model.add(MaxPool1D())

model.add(LSTM(units=128))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])



del embedding_matrix
model.summary()
model.fit(X_train, y_train, validation_split=0.3, epochs=12)
y_pred = (model.predict(X_test) > 0.5).astype("int")
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))