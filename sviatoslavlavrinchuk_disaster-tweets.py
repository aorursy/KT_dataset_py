import pandas as pd

import matplotlib

import numpy as np

import re

import nltk

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop, Adam

from keras.callbacks import EarlyStopping

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords

from keras import backend as K

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import GridSearchCV 

from sklearn.impute import KNNImputer

import category_encoders as ce

from sklearn.model_selection import cross_val_score

import langid

pd.set_option('display.max_rows', 1000)

from keras.preprocessing.text import Tokenizer
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')

dfs = [df_train, df_test]

df = df_train.copy() # just to fill free to mess up the data a bit:)

df
import re

from nltk.stem import SnowballStemmer

from nltk import WordNetLemmatizer

import nltk

wnl = WordNetLemmatizer()

from nltk.corpus import stopwords, wordnet

nltk.download('wordnet')
def cleanSentences(text, remove_stopwords=True, stem_words=True):

    # Clean the text, with the option to remove stopwords and to stem words.

    

    # Convert words to lower case and split them

    text = text.lower().replace("<br />", " ")

    text = text.split()



    # Optionally, remove stop words

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        text = [w for w in text if not w in stops]

    

    text = " ".join(text)



    # Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    

    def get_wordnet_pos(tag):

        if tag.startswith('J'):

            return wordnet.ADJ

        elif tag.startswith('V'):

            return wordnet.VERB

        elif tag.startswith('N'):

            return wordnet.NOUN

        elif tag.startswith('R'):

            return wordnet.ADV

        else:

            return wordnet.NOUN

    text = text.split()

    text = nltk.tag.pos_tag(text)

    text = [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in text]

    text = [wnl.lemmatize(word, tag) for word, tag in text]

    text = ' '.join(text)

    # Optionally, shorten words to their stems

#     if stem_words:

#         text = text.split()

#         stemmer = SnowballStemmer('english')

#         stemmed_words = [stemmer.stem(word) for word in text]

#         text = " ".join(stemmed_words)

    

    # Return a list of words

    return(text)
df["clean"] = df.text.apply(lambda x: cleanSentences(x))
from sklearn.model_selection import train_test_split

X = np.array(df.clean)



Y = np.array(df.target)



Y = Y.reshape(-1,1)

print(Y.shape)

# Y_reg = Y_reg.reshape(-1,1)





X_train, X_test, Y_train, Y_test,  = train_test_split(X, Y, test_size=0.1, random_state = 42)
MAX_WORDS = 250000

EMBEDDING_DIM = 300

tok = Tokenizer(num_words=MAX_WORDS)

tok.fit_on_texts(X)

sequences = tok.texts_to_sequences(X_train)

word_index = tok.word_index

print('Found %s unique tokens' % len(word_index))

MAX_SEQUENCE_LENGTH = 300

sequences_matrix = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print(sequences_matrix)

print('Shape of data tensor:', sequences_matrix.shape)
nb_words = min(MAX_WORDS, len(word_index))+1

print(nb_words)
import spacy



nlp = spacy.load('en_core_web_lg')


embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

print(embedding_matrix.shape)

for word, i in word_index.items():

    embedding_matrix[i] = np.array(nlp(word)[0].vector)

#     if word in word2vec.vocab:

#         embedding_matrix[i] = Word2Vec(word, min_count=2)

#         embedding_matrix[i] = word2vec.word_vec(word)

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

print(embedding_matrix.shape)
def RNN():

    inputs = Input(name='inputs',shape=[MAX_SEQUENCE_LENGTH])

    layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix],

                      input_length=MAX_SEQUENCE_LENGTH, trainable=False)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(256,name='FullyConnectedCls1')(layer)

    layer = Activation('relu')(layer)

    layer_branching = Dropout(0.5)(layer)

    

    

    

    layer = Dense(64,name='FullyConnectedCls2')(layer_branching)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    

    layer = Dense(1,name='out_layer_cls')(layer)

    layer_out_cls = Activation('sigmoid',name='ActivationCls2')(layer)

    model = Model(inputs=inputs, outputs=layer_out_cls)

    

    

    return model
model = RNN()

# model.summary()



model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00005), metrics=['acc'])
(trainX, testX, trainY, testY) = train_test_split(sequences_matrix, Y_train, test_size=0.2, random_state=42)



hist = model.fit(trainX,

                  trainY, batch_size=100,

                 epochs=22,

                 validation_data=(testX, testY)

                )

matplotlib.pyplot.plot(hist.history['acc'])

matplotlib.pyplot.plot(hist.history['val_acc'])

matplotlib.pyplot.title('Classification model accuracy')

matplotlib.pyplot.ylabel('accuracy')

matplotlib.pyplot.xlabel('epoch')

matplotlib.pyplot.legend(['train', 'test'], loc='upper left')

matplotlib.pyplot.show()
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)

accr = model.evaluate(test_sequences_matrix,Y_test)

print("test loss, test acc:", accr)

prediction = model.predict(test_sequences_matrix)

prediction_recovered = np.round(prediction * 10)



binary_prediction = [0 if item[0]<0.5 else 1 for item in prediction]

df_tesset = pd.DataFrame()

df_tesset['text'] = X_test

df_tesset['binary'] = binary_prediction

df_tesset['label'] = Y_test
df_test["clean"] = df_test.text.apply(lambda x: cleanSentences(x))

X_test = df_test["clean"]
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)

# accr = model.evaluate(test_sequences_matrix,Y_test)

# print("test loss, test acc:", accr)

prediction = model.predict(test_sequences_matrix)

prediction_recovered = np.round(prediction * 10)



binary_prediction = [0 if item[0]<0.5 else 1 for item in prediction]

df_tesset = pd.DataFrame()

df_tesset['id'] = df_test.id

df_tesset['target'] = binary_prediction

df_tesset.to_csv ('submission.csv', index = False, header=True, sep = ",")
df_tesset = pd.DataFrame()

df_tesset['text'] = X_test

df_tesset['binary'] = binary_prediction

df_tesset['label'] = Y_test
df_tesset
len(df_tesset.loc[df_tesset.binary == df_tesset.label])/len(df_tesset)
df["lang"]  = df["clean"].apply(lambda x: langid.classify(x)[0])
df.lang.value_counts()
df.loc[df.lang == "de"]
df.location.value_counts()
groups = df.groupby([df.keyword,df.target]).count()

groups.id
fig = px.bar(groups, x = "id")

fig.show()
sns.countplot(df.target)