# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from numpy import array, asarray, zeros
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
from nltk.stem import PorterStemmer
stemming = PorterStemmer()
fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
embedding = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')

fake.head()
fake = fake.drop(['title','subject','date'], axis=1)
fake['text'].replace(' ', np.nan, inplace=True)
print(fake.count())
fake.dropna(subset = ["text"], inplace=True)
print(fake.count())
fake['validity']=0
fake.head()
true.head()
true = true.drop(['title','subject','date'], axis=1)
true['text'].replace(' ', np.nan, inplace=True)
print(true.count())
true.dropna(subset = ["text"], inplace=True)
print(true.count())
true['validity']= 1
true.head()
data = pd.concat([fake,true], ignore_index=True, sort=False)
data.head()
data.describe()
data.isnull().sum()
sns.countplot(x='validity', data=data)
def clean_data(sen):

    # Removing html tags
    sentence = re.sub(r'<[^>]+>', ' ', sen)
    
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Removing single quote
    sentence = re.sub("'", ' ', sentence)
    
    #strip out non alphanumeric words/characters (such as numbers and punctuation) using .isalpha
    tokens = nltk.word_tokenize(sentence)
    # taken only words (not punctuation)
    sentence = [w for w in tokens if w.isalpha()]
    
    #Stemming reduces related words to a common stem
    sentence = [stemming.stem(word) for word in sentence]
    
    #Removing stop words
    sentence = [w for w in sentence if not w in stops]
    
    #Rejoin words to sentence
    sentence = ( " ".join(sentence))

    return sentence
X = []
sentences = list(data.text)
for sen in sentences:
    X.append(clean_data(sen))
#X
from wordcloud import WordCloud
merge_all_sentences= ' '.join(X)
wordcloud = WordCloud().generate(merge_all_sentences)
pyplot.imshow(wordcloud, interpolation='bilinear')
pyplot.axis("off")
y = data['validity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1
maxlen = 500

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
embeddings_dictionary = dict()
#print(embedding)
for line in embedding:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions

EMBEDDING_DIM = 100
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    #print(embedding_vector)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
model = Sequential()

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
fit_model = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.25)
from sklearn.metrics import classification_report, accuracy_score
y_pred = (model.predict(X_test) > 0.5).astype("int")
print(classification_report(y_test, y_pred))
