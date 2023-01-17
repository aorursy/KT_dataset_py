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
# Imports statements.

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re



#Data Preprocessing and Feature Engineering

from wordcloud import WordCloud, STOPWORDS 

from gensim.parsing.preprocessing import remove_stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



#Model Selection and Validation

from sklearn.naive_bayes import MultinomialNB

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

%matplotlib inline
def display_all_details(dataframe):

    print(('='*50)+'DATA'+('='*50))

    print(('-'*50)+'SHAPE'+('-'*50))

    print(dataframe.shape)

    print(('-'*50)+'COLUMNS'+('-'*50))

    print(dataframe.columns)

    print(('-'*50)+'DESCRIBE'+('-'*50))

    print(dataframe.describe())

    print(('-'*50)+'INFO'+('-'*50))

    print(dataframe.info())

    print(('='*50)+'===='+('='*50))
twitter_data = pd.read_csv('../input/twitterdata/finalSentimentdata2.csv')
display_all_details(twitter_data)
twitter_data.head()
twitter_data.tail()
# Checking missing values columns

twitter_data.isnull().sum()
print(twitter_data.sentiment.value_counts())

twitter_data.sentiment.value_counts().plot(kind = 'bar')
# Creating a mapping for sentiments

mapping = {'fear':0,

          'sad':1,

          'anger':2,

          'joy':3}
twitter_data['sentiment'] = twitter_data['sentiment'].map(mapping)
twitter_data.head()
for tweet in twitter_data.text.head(20):

    print(tweet)
def clean_text_column(row):

    text = row['text'].lower()

    text = re.sub(r'[^(a-zA-Z\s)]','',text)

    text = re.sub(r'\(','',text)

    text = re.sub(r'\)','',text)

    text = text.replace('\n',' ')

    text = text.strip()

    return text
twitter_data['cleaned_text'] = twitter_data.apply(clean_text_column,axis = 1)
twitter_data.head()
# These are new stopwords which i add after several model runs and found out these are irrelevant words which are created which cleaning process.

new_additions=['aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']
new_string = ''

stopwords = set(list(STOPWORDS)+new_additions)

for val in twitter_data.cleaned_text: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    new_string += " ".join(tokens)+" "
wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(new_string) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
# Check for spaced entry which can be created due to cleaning step.

twitter_data.cleaned_text.str.isspace().sum()
filtered_sentences = []

for tweet in twitter_data.cleaned_text:

    filtered_sentences.append(remove_stopwords(tweet))
filter_sentence_df = pd.DataFrame(filtered_sentences,columns = ['filter_sentence'])
new_twitter_data = pd.concat([twitter_data,filter_sentence_df],axis = 1)
new_twitter_data.head()
#Normalizing the words in tweets 

def normalization(tweet):

    lem = WordNetLemmatizer()

    normalized_tweet = []

    for word in tweet['filter_sentence'].split():

        normalized_text = lem.lemmatize(word,'v')

        normalized_tweet.append(normalized_text)

    return normalized_tweet
new_twitter_data['normalised_tweet'] = new_twitter_data.apply(normalization,axis = 1)
new_twitter_data.head()
msg_train, msg_test, label_train, label_test = train_test_split(new_twitter_data['filter_sentence'],new_twitter_data['sentiment'], test_size=0.1,random_state = 2)
pipeline = Pipeline([

    ('tfidf', TfidfVectorizer(stop_words=stopwords)),

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

    ])
pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(predictions,label_test))

print(confusion_matrix(predictions,label_test))

print(accuracy_score(predictions,label_test))
pipeline2 = Pipeline([

    ('tfidf', TfidfVectorizer(stop_words=stopwords)),

    ('classifier',LogisticRegression(solver='sag')),  # train on TF-IDF vectors w/ Naive Bayes classifier

    ])
pipeline2.fit(msg_train,label_train)

predictions2 = pipeline2.predict(msg_test)

print(classification_report(predictions2,label_test))

print(confusion_matrix(predictions2,label_test))

print(accuracy_score(predictions2,label_test))
pipeline3 = Pipeline([

                ('tfidf', TfidfVectorizer(stop_words=stopwords)),

                ('clf', OneVsRestClassifier(SVC(), n_jobs=1)),

            ])
pipeline3.fit(msg_train,label_train)

predictions3 = pipeline3.predict(msg_test)

print(classification_report(predictions3,label_test))

print(confusion_matrix(predictions3,label_test))

print(accuracy_score(predictions3,label_test))
from sklearn.ensemble import VotingClassifier
voting_classifier = VotingClassifier(estimators=[ ('nb', pipeline),('lr', pipeline2), ('svc', pipeline3)], voting='hard')
voting_classifier.fit(msg_train,label_train)

predictions4 = voting_classifier.predict(msg_test)

print(classification_report(predictions4,label_test))

print(confusion_matrix(predictions4,label_test))

print(accuracy_score(predictions4,label_test))
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from keras.callbacks import EarlyStopping

from keras.layers import Dropout
new_twitter_data.head()
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 1000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 250

# This is fixed.

EMBEDDING_DIM = 100



tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(new_twitter_data.filter_sentence.values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(new_twitter_data.filter_sentence.values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)
Y = pd.get_dummies(new_twitter_data.sentiment).values

print('Shape of label tensor:', Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.1))

model.add(LSTM(100, dropout=0.1, recurrent_dropout=0.1))

model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
epochs = 20

batch_size = 64

# I am using EarlyStopping to monitor val_loss upto 3 patience level to prevent the model from overfitting.

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
accr = model.evaluate(X_test,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
plt.title('Accuracy')

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.show()