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
dfTrue = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

dfFake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
dfTrue.head()
dfFake.head()
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline



stpwords = set(STOPWORDS)
import re

# generate real word list

realNewsWords = [str(i) for i in dfTrue['title']]

realWordsString = (" ".join(realNewsWords)).lower()

realWordsString = re.sub(r'[^\w\s]', '', realWordsString)
# Generate Word cloud

wc = WordCloud(width = 800, height = 800,

               stopwords = stpwords,

              background_color = 'white').generate(realWordsString)

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wc); 

plt.axis("off");
# Generate Fake word list

fakeNewsWords = [str(i) for i in dfFake['title']]

fakeWordsString = (" ".join(fakeNewsWords)).lower()

fakeWordsString = re.sub(r'[^\w\s]', '', fakeWordsString)



# generate Word cloud

wc = WordCloud(width = 800, height = 800,

               stopwords = stpwords,

              background_color = 'white').generate(fakeWordsString)

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wc); 

plt.axis("off");
# get the column names

dfFake.columns
# Transform the  Fake dataset

fakeData = dfFake.drop(['text', 'subject', 'date'], axis = 1)

fakeData['Prediction'] = pd.Series([0]*len(fakeData))

fakeData.head()
# Transform the real dataset

realData = dfTrue.drop(['text', 'subject', 'date'], axis = 1)

realData['Prediction'] = pd.Series([1]*len(realData))

realData.head()
# generate the data

data = pd.concat([realData, fakeData], axis = 0, ignore_index = True)
# top of data has real news

data.head()
# bottom has fake news

data.tail()
import nltk

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

nltk.download('stopwords')
### cleans the text

def clean_text(text):

    ''' Cleans the text data, then returns a list of words'''

    ps = PorterStemmer()  # Stemmer

    clean_text = text.lower()  # make all into lower case

    clean_text = re.sub('[^A-Za-z\s]+', ' ', clean_text) # remove punctuations and numbers

    clean_text = clean_text.split() # list of words

    clean_text = [ps.stem(word) for word in clean_text if not word in stopwords.words('english')] # Stopword removal

    clean_text = ' '.join(clean_text)

    

    return clean_text
clean_text("this is 43 ? i though DONAld troops is missing values sunshine!")
X = data.iloc[ : , :-1]

y = data.iloc[ : , -1]

y.head()  # Separating into dependant and independent features

X.head()
import tensorflow as tf

from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense
newsCorpus = [clean_text(text) for text in X['title']]
len(newsCorpus)
# Defined vocabulary size

vocabSize = 6000
# One hot representation of all the news articles

one_hot_news = [one_hot(text, vocabSize) for text in newsCorpus]

one_hot_news[:5]
# Get max length

max([len(vec) for vec in one_hot_news])
# Pad the sentences, make fixed length

max_length = 40

embedded_news = pad_sequences(one_hot_news, padding = 'pre', maxlen = max_length)

embedded_news[:5]
embedding_features_length = 40

from tensorflow.keras.layers import Dropout

model = Sequential()

model.add(Embedding(vocabSize, embedding_features_length, input_length = max_length))

model.add(Dropout(0.4))

model.add(LSTM(120))

model.add(Dropout(0.4))

model.add(Dense(1, activation = 'sigmoid'))
#Compile the model

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(model.summary())
# Train test Split

from sklearn.model_selection import train_test_split



X = np.array(embedded_news)

y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Fit the model

# Now we fit the model

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 5, batch_size = 64)
y_pred = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test, y_pred)

import seaborn as sns

sns.heatmap(data = mat, annot = True)