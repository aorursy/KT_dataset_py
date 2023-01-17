import pandas as pd

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import roc_curve, auc

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix 

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from nltk.stem.porter import PorterStemmer
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
test.head()
train.info()

test.info()

train.describe()
df = train.append(test,ignore_index=True)

df.head()
df.info()

# Adding the word counts to a dataframe is a very good practice because we might use these counts to reach some useful information.





# fill NA values by space

df['Review Text'] = df['Review Text'].fillna('')



# CountVectorizer() converts a collection 

# of text documents to a matrix of token counts

vectorizer = CountVectorizer()

# assign a shorter name for the analyze

# which tokenizes the string

analyzer = vectorizer.build_analyzer()



def wordcounts(s):

    c = {}

    # tokenize the string and continue, if it is not empty

    if analyzer(s):

        d = {}

        # find counts of the vocabularies and transform to array 

        w = vectorizer.fit_transform([s]).toarray()

        # vocabulary and index (index of w)

        vc = vectorizer.vocabulary_

        # items() transforms the dictionary's (word, index) tuple pairs

        for k,v in vc.items():

            d[v]=k # d -> index:word 

        for index,i in enumerate(w[0]):

            c[d[index]] = i # c -> word:count

    return  c



# add new column to the dataframe

df['Word Counts'] = df['Review Text'].apply(wordcounts)

df.head()

#Checking How is the frequency of words in the review text

rt = df['Review Text']

plt.subplots(figsize=(18,6))

wordcloud = WordCloud(background_color='black',

                      width=900,

                      height=300

                     ).generate(" ".join(rt))

plt.imshow(wordcloud)

plt.title('All Words in the Reviews\n',size=25)

plt.axis('off')

plt.show()
ps = PorterStemmer()

Reviews = df['Review Text'].astype(str)

print(Reviews.shape)

Reviews[Reviews.isnull()] = "NULL"
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')

stop_words = set(stopwords.words('english'))

def preprocessing(data):

    txt = data.str.lower().str.cat(sep=' ') #1

    words = tokenizer.tokenize(txt) #2

    words = [w for w in words if not w in stop_words] #3

    #words = [ps.stem(w) for w in words] #4

    return words
df['tokenized'] = df["Review Text"].astype(str).str.lower() # Turn into lower case text

df['tokenized'] = df.apply(lambda row: tokenizer.tokenize(row['tokenized']), axis=1) # Apply tokenize to each row

df['tokenized'] = df['tokenized'].apply(lambda x: [w for w in x if not w in stop_words]) # Remove stopwords from each row
def string_unlist(strlist):

    return " ".join(strlist)



df["tokenized_unlist"] = df["tokenized"].apply(string_unlist)

df.head()
import statsmodels.api as sm

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Pre-Processing

SIA = SentimentIntensityAnalyzer()



# Applying Model, Variable Creation

df['Polarity Score']=df["tokenized_unlist"].apply(lambda x:SIA.polarity_scores(x)['compound'])

df['Neutral Score']=df["tokenized_unlist"].apply(lambda x:SIA.polarity_scores(x)['neu'])

df['Negative Score']=df["tokenized_unlist"].apply(lambda x:SIA.polarity_scores(x)['neg'])

df['Positive Score']=df["tokenized_unlist"].apply(lambda x:SIA.polarity_scores(x)['pos'])



# Converting 0 to 1 Decimal Score to a Categorical Variable

df['Sentiment']=''

df.loc[df['Polarity Score']>0,'Sentiment']='Positive'

df.loc[df['Polarity Score']==0,'Sentiment']='Neutral'

df.loc[df['Polarity Score']<0,'Sentiment']='Negative'

conditions = [

    df['Sentiment'] == "Positive",

    df['Sentiment'] == "Negative",

    df['Sentiment'] == "Neutral"]

choices = [1,-1,0]

df['label'] = np.select(conditions, choices)

df.head()
#Simple Embedding Deep Neural Network



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense, SimpleRNN



samples = df["tokenized_unlist"].tolist()

maxlen = 100 

max_words = 5000

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(df["label"].values)

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)
indices = np.arange(df.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]
training_samples = 4043

validation_samples = 5014

x_train = data[:training_samples]

y_train = labels[:training_samples]

x_val = data[training_samples: validation_samples] 

y_val = labels[training_samples: validation_samples]

x_test = data[validation_samples:]

y_test = labels[validation_samples:]

x_train = pad_sequences(x_train, maxlen=maxlen)

x_val = pad_sequences(x_val, maxlen=maxlen)
# BASELINE

# That is, if all the labels are predicted as 1

(np.sum(df['label'] == 1)/df.shape[0]) * 100



# we have to make model that performs better than this baseline

def build_model():

    model = Sequential()

    model.add(Embedding(max_words, 100, input_length=maxlen))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['acc'])

    return model

model = build_model()

model.summary()

history = model.fit(x_train, y_train,

                    epochs=15,

                    batch_size=16,

                    validation_data=(x_val, y_val))



model.save("model1.h5")
res = model.predict(x_test)
res
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()


def build_RNN():

    model = Sequential() 

    model.add(Embedding(max_words, 100, input_length=maxlen)) 

    #model.add(SimpleRNN(32, return_sequences=True))

    model.add(SimpleRNN(32)) 

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 

    return model

model = build_RNN()

model.summary()

history_RNN = model.fit(x_train, y_train,

                    epochs=5,

                    batch_size=32,

                    validation_data=(x_val, y_val))



model.save("model_RNN.h5")

acc = history_RNN.history['acc']

val_acc = history_RNN.history['val_acc']

loss = history_RNN.history['loss']

val_loss = history_RNN.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
model.evaluate(x_test, y_test)

