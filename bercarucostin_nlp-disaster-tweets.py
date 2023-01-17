# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



from nltk.corpus import stopwords    

from nltk.tokenize import word_tokenize

from textblob import TextBlob



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, MaxPooling1D, Conv1D, Concatenate, Bidirectional, GlobalMaxPool1D, ActivityRegularization, BatchNormalization

from keras.models import Model

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score



from tqdm import tqdm



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import re

import string



%matplotlib inline

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train_df.head()
train_df['location'].value_counts().iplot(kind = 'bar')

train_df['haslocation'] = np.where(train_df['location'].isnull(), 0, 1)

"{:.2f}% of the entries have a location".format((len(train_df[(train_df['haslocation'] == 1)]) / len(train_df) * 100))
"{:.2f}% of the entries which represent actual disasters have a location".format((len(train_df[(train_df['haslocation'] == 1) & (train_df['target'] == 1)]) / len(train_df[(train_df['target'] == 1)]) * 100)) 
"{:.2f}% of the entries which do not represent disasters have a location".format((len(train_df[(train_df['haslocation'] == 1) & (train_df['target'] == 0)]) / len(train_df[(train_df['target'] == 0)]) * 100)) 
train_df.drop(["location", "haslocation"], axis = 1, inplace = True)

test_df.drop(["location"], axis = 1, inplace = True)
train_df['keyword'].value_counts().iplot(kind = 'bar')

train_df['haskeyword'] = np.where(train_df['keyword'].isnull(), 0, 1)

"{:.2f}% of the entries have a keyword".format((len(train_df[(train_df['haskeyword'] == 1)]) / len(train_df) * 100)) 
train_df.where(train_df['target'] == 1)['keyword'].value_counts().iplot(kind = 'bar')

train_df.where(train_df['target'] == 0)['keyword'].value_counts().iplot(kind = 'bar')
train_df['keyword_sentiment'] = train_df['keyword'].apply(lambda x: (TextBlob(x).sentiment[0] + 1) / 2 if type(x) == str else None)

train_df['tweet_sentiment'] = train_df['text'].apply(lambda x: (TextBlob(x).sentiment[0] + 1) / 2)

test_df['tweet_sentiment'] = test_df['text'].apply(lambda x: (TextBlob(x).sentiment[0] + 1) / 2)
train_df[['target', 'keyword_sentiment']].pivot(columns = 'target', values = 'keyword_sentiment').iplot(kind = 'box')

train_df[['target', 'tweet_sentiment']].pivot(columns = 'target', values = 'tweet_sentiment').iplot(kind = 'box')
train_df['text'] = train_df['keyword'].astype(str) + ' ' + train_df['text'].astype(str)

test_df['text'] = test_df['keyword'].astype(str) + ' ' + test_df['text'].astype(str)



train_df.drop(["keyword", "haskeyword", "keyword_sentiment"], axis = 1, inplace = True)

test_df.drop(["keyword"], axis = 1, inplace = True)
train_df['nr_of_char'] = train_df['text'].str.len()

train_df['nr_of_char'] = train_df['nr_of_char'] / train_df['nr_of_char'].max()

train_df[['target', 'nr_of_char']].pivot(columns = 'target', values = 'nr_of_char').iplot(kind = 'box')



test_df['nr_of_char'] = test_df['text'].str.len()

test_df['nr_of_char'] = test_df['nr_of_char'] / test_df['nr_of_char'].max()
train_df['nr_of_words'] = train_df['text'].str.split().str.len()

train_df['nr_of_words'] = train_df['nr_of_words'] / train_df['nr_of_words'].max()

train_df[['target', 'nr_of_words']].pivot(columns = 'target', values = 'nr_of_words').iplot(kind = 'box')



test_df['nr_of_words'] = test_df['text'].str.split().str.len()

test_df['nr_of_words'] = test_df['nr_of_words'] / test_df['nr_of_words'].max()
train_df['nr_of_unique_words'] = train_df['text'].apply(lambda x: len(set(x.split())))

train_df['nr_of_unique_words'] = train_df['nr_of_unique_words'] / train_df['nr_of_unique_words'].max()

train_df[['target', 'nr_of_unique_words']].pivot(columns = 'target', values = 'nr_of_unique_words').iplot(kind='box')



test_df['nr_of_unique_words'] = test_df['text'].apply(lambda x: len(set(x.split())))

test_df['nr_of_unique_words'] = test_df['nr_of_unique_words'] / test_df['nr_of_unique_words'].max()
train_df['nr_of_punctuation'] = train_df['text'].str.split(r"\?|,|\.|\!|\"|'").str.len()

train_df['nr_of_punctuation'] = train_df['nr_of_punctuation'] / train_df['nr_of_punctuation'].max()

train_df[['target', 'nr_of_punctuation']].pivot(columns = 'target', values = 'nr_of_punctuation').iplot(kind = 'box')



test_df['nr_of_punctuation'] = test_df['text'].str.split(r"\?|,|\.|\!|\"|'").str.len()

test_df['nr_of_punctuation'] = test_df['nr_of_punctuation'] / test_df['nr_of_punctuation'].max()
stop_words = set(stopwords.words('english'))

train_df['nr_of_stopwords'] = train_df['text'].str.split().apply(lambda x: len(set(x) & stop_words))

train_df['nr_of_stopwords'] = train_df['nr_of_stopwords'] / train_df['nr_of_stopwords'].max()

train_df[['target', 'nr_of_stopwords']].pivot(columns = 'target', values = 'nr_of_stopwords').iplot(kind = 'box')



test_df['nr_of_stopwords'] = test_df['text'].str.split().apply(lambda x: len(set(x) & stop_words))

test_df['nr_of_stopwords'] = test_df['nr_of_stopwords'] / test_df['nr_of_stopwords'].max()
train_df.corr().iplot(kind='heatmap',colorscale="Reds",title="Feature Correlation Matrix")
glove_embeddings = np.load('/kaggle/input/embfile/emb/glove.840B.300d.pkl', allow_pickle=True)



def build_vocab(X):

    

    tweets = X.apply(lambda s: s.split()).values      

    vocab = {}

    

    for tweet in tweets:

        for word in tweet:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1                

    return vocab



def check_embeddings_coverage(X, embeddings):

    

    vocab = build_vocab(X)    

    

    covered = {}

    oov = {}    

    n_covered = 0

    n_oov = 0

    

    for word in vocab:

        try:

            covered[word] = embeddings[word]

            n_covered += vocab[word]

        except:

            n_oov += vocab[word]

            

    vocab_coverage = len(covered) / len(vocab)

    text_coverage = (n_covered / (n_covered + n_oov))

    

    return vocab_coverage, text_coverage



train_glove_vocab_coverage, train_glove_text_coverage = check_embeddings_coverage(train_df['text'], glove_embeddings)

print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_glove_vocab_coverage, train_glove_text_coverage))

test_glove_vocab_coverage, test_glove_text_coverage = check_embeddings_coverage(test_df['text'], glove_embeddings)

print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_glove_vocab_coverage, test_glove_text_coverage))
def clean(tweet): 

            

    # Special characters

    tweet = re.sub(r"\x89Û_", "", tweet)

    tweet = re.sub(r"\x89ÛÒ", "", tweet)

    tweet = re.sub(r"\x89ÛÓ", "", tweet)

    tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)

    tweet = re.sub(r"\x89ÛÏ", "", tweet)

    tweet = re.sub(r"China\x89Ûªs", "China's", tweet)

    tweet = re.sub(r"let\x89Ûªs", "let's", tweet)

    tweet = re.sub(r"\x89Û÷", "", tweet)

    tweet = re.sub(r"\x89Ûª", "", tweet)

    tweet = re.sub(r"\x89Û\x9d", "", tweet)

    tweet = re.sub(r"å_", "", tweet)

    tweet = re.sub(r"\x89Û¢", "", tweet)

    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)

    tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)

    tweet = re.sub(r"åÊ", "", tweet)

    tweet = re.sub(r"åÈ", "", tweet)

    tweet = re.sub(r"JapÌ_n", "Japan", tweet)    

    tweet = re.sub(r"Ì©", "e", tweet)

    tweet = re.sub(r"å¨", "", tweet)

    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)

    tweet = re.sub(r"åÇ", "", tweet)

    tweet = re.sub(r"å£3million", "3 million", tweet)

    tweet = re.sub(r"åÀ", "", tweet)

    

    # Contractions

    tweet = re.sub(r"he's", "he is", tweet)

    tweet = re.sub(r"there's", "there is", tweet)

    tweet = re.sub(r"We're", "We are", tweet)

    tweet = re.sub(r"That's", "That is", tweet)

    tweet = re.sub(r"won't", "will not", tweet)

    tweet = re.sub(r"they're", "they are", tweet)

    tweet = re.sub(r"Can't", "Cannot", tweet)

    tweet = re.sub(r"wasn't", "was not", tweet)

    tweet = re.sub(r"don\x89Ûªt", "do not", tweet)

    tweet = re.sub(r"aren't", "are not", tweet)

    tweet = re.sub(r"isn't", "is not", tweet)

    tweet = re.sub(r"What's", "What is", tweet)

    tweet = re.sub(r"haven't", "have not", tweet)

    tweet = re.sub(r"hasn't", "has not", tweet)

    tweet = re.sub(r"There's", "There is", tweet)

    tweet = re.sub(r"He's", "He is", tweet)

    tweet = re.sub(r"It's", "It is", tweet)

    tweet = re.sub(r"You're", "You are", tweet)

    tweet = re.sub(r"I'M", "I am", tweet)

    tweet = re.sub(r"shouldn't", "should not", tweet)

    tweet = re.sub(r"wouldn't", "would not", tweet)

    tweet = re.sub(r"i'm", "I am", tweet)

    tweet = re.sub(r"I\x89Ûªm", "I am", tweet)

    tweet = re.sub(r"I'm", "I am", tweet)

    tweet = re.sub(r"Isn't", "is not", tweet)

    tweet = re.sub(r"Here's", "Here is", tweet)

    tweet = re.sub(r"you've", "you have", tweet)

    tweet = re.sub(r"you\x89Ûªve", "you have", tweet)

    tweet = re.sub(r"we're", "we are", tweet)

    tweet = re.sub(r"what's", "what is", tweet)

    tweet = re.sub(r"couldn't", "could not", tweet)

    tweet = re.sub(r"we've", "we have", tweet)

    tweet = re.sub(r"it\x89Ûªs", "it is", tweet)

    tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)

    tweet = re.sub(r"It\x89Ûªs", "It is", tweet)

    tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)

    tweet = re.sub(r"who's", "who is", tweet)

    tweet = re.sub(r"I\x89Ûªve", "I have", tweet)

    tweet = re.sub(r"y'all", "you all", tweet)

    tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)

    tweet = re.sub(r"would've", "would have", tweet)

    tweet = re.sub(r"it'll", "it will", tweet)

    tweet = re.sub(r"we'll", "we will", tweet)

    tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)

    tweet = re.sub(r"We've", "We have", tweet)

    tweet = re.sub(r"he'll", "he will", tweet)

    tweet = re.sub(r"Y'all", "You all", tweet)

    tweet = re.sub(r"Weren't", "Were not", tweet)

    tweet = re.sub(r"Didn't", "Did not", tweet)

    tweet = re.sub(r"they'll", "they will", tweet)

    tweet = re.sub(r"they'd", "they would", tweet)

    tweet = re.sub(r"DON'T", "DO NOT", tweet)

    tweet = re.sub(r"That\x89Ûªs", "That is", tweet)

    tweet = re.sub(r"they've", "they have", tweet)

    tweet = re.sub(r"i'd", "I would", tweet)

    tweet = re.sub(r"should've", "should have", tweet)

    tweet = re.sub(r"You\x89Ûªre", "You are", tweet)

    tweet = re.sub(r"where's", "where is", tweet)

    tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)

    tweet = re.sub(r"we'd", "we would", tweet)

    tweet = re.sub(r"i'll", "I will", tweet)

    tweet = re.sub(r"weren't", "were not", tweet)

    tweet = re.sub(r"They're", "They are", tweet)

    tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)

    tweet = re.sub(r"you\x89Ûªll", "you will", tweet)

    tweet = re.sub(r"I\x89Ûªd", "I would", tweet)

    tweet = re.sub(r"let's", "let us", tweet)

    tweet = re.sub(r"it's", "it is", tweet)

    tweet = re.sub(r"can't", "cannot", tweet)

    tweet = re.sub(r"don't", "do not", tweet)

    tweet = re.sub(r"you're", "you are", tweet)

    tweet = re.sub(r"i've", "I have", tweet)

    tweet = re.sub(r"that's", "that is", tweet)

    tweet = re.sub(r"i'll", "I will", tweet)

    tweet = re.sub(r"doesn't", "does not", tweet)

    tweet = re.sub(r"i'd", "I would", tweet)

    tweet = re.sub(r"didn't", "did not", tweet)

    tweet = re.sub(r"ain't", "am not", tweet)

    tweet = re.sub(r"you'll", "you will", tweet)

    tweet = re.sub(r"I've", "I have", tweet)

    tweet = re.sub(r"Don't", "do not", tweet)

    tweet = re.sub(r"I'll", "I will", tweet)

    tweet = re.sub(r"I'd", "I would", tweet)

    tweet = re.sub(r"Let's", "Let us", tweet)

    tweet = re.sub(r"you'd", "You would", tweet)

    tweet = re.sub(r"It's", "It is", tweet)

    tweet = re.sub(r"Ain't", "am not", tweet)

    tweet = re.sub(r"Haven't", "Have not", tweet)

    tweet = re.sub(r"Could've", "Could have", tweet)

    tweet = re.sub(r"youve", "you have", tweet)  

    tweet = re.sub(r"donå«t", "do not", tweet)   

           

    # Urls

    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)

        

    # Words with punctuations and special characters

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    for p in punctuations:

        tweet = tweet.replace(p, '')

        

    # ... and ..

    tweet = tweet.replace('...', ' ... ')

    if '...' not in tweet:

        tweet = tweet.replace('..', ' ... ')

        

    #Spaces

    tweet = tweet.replace('  ', ' ')

    tweet = tweet.replace('   ', ' ')

        

    tweet = tweet.lower()

    

    tweet = " ".join(tweet.split())

    

    return tweet



train_df['text_cleaned'] = train_df['text'].apply(lambda s : clean(s))

test_df['text_cleaned'] = test_df['text'].apply(lambda s : clean(s))



train_glove_vocab_coverage, train_glove_text_coverage = check_embeddings_coverage(train_df['text_cleaned'], glove_embeddings)

print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Training Set'.format(train_glove_vocab_coverage, train_glove_text_coverage))

test_glove_vocab_coverage, test_glove_text_coverage = check_embeddings_coverage(test_df['text_cleaned'], glove_embeddings)

print('GloVe Embeddings cover {:.2%} of vocabulary and {:.2%} of text in Test Set'.format(test_glove_vocab_coverage, test_glove_text_coverage))
def create_corpus(df):

    

    corpus = []

    

    for tweet in tqdm(df['text_cleaned']):

        words = [word.lower() for word in word_tokenize(tweet) if((word.isalpha() == 1) and word not in stop_words)]

        corpus.append(words)

        

    return corpus



train_corpus = create_corpus(train_df)

test_corpus = create_corpus(test_df)



tokenizer_obj_train = Tokenizer()

tokenizer_obj_train.fit_on_texts(train_corpus)

seq_train=tokenizer_obj_train.texts_to_sequences(train_corpus)



tokenizer_obj_test = Tokenizer()

tokenizer_obj_test.fit_on_texts(test_corpus)

seq_test=tokenizer_obj_test.texts_to_sequences(test_corpus)



MAX_WORDS = 0

for i in seq_train:

    MAX_WORDS = max(MAX_WORDS, len(i))



for i in seq_test:

    MAX_WORDS = max(MAX_WORDS, len(i))

    

train_pad = pad_sequences(seq_train, maxlen = MAX_WORDS, truncating = 'post', padding = 'post')

test_pad = pad_sequences(seq_test, maxlen = MAX_WORDS, truncating = 'post', padding = 'post')



tokenizer_obj_train.word_index.update(tokenizer_obj_test.word_index)
word_index = tokenizer_obj_train.word_index

print('Number of unique words:',len(word_index))
num_words = len(word_index) + 1

embedding_matrix = np.zeros((num_words, 300))



counter = 0



for word, i in tqdm(word_index.items()):

    emb_vec = glove_embeddings.get(word)

    if emb_vec is not None:

        embedding_matrix[i] = emb_vec

    else:

        counter += 1



del glove_embeddings

counter
x_train, x_test, y_train, y_test = train_test_split(train_pad, train_df['target'].values, test_size = 0.25, random_state = 42)



print("Shape of train", x_train.shape)

print("Shape of Validation", x_test.shape)
inp = Input(shape = (MAX_WORDS, ))

x = Embedding(num_words, 300, weights = [embedding_matrix])(inp)

x = Bidirectional(LSTM(MAX_WORDS, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))(x)

x = ActivityRegularization(l2 = 0.1)(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.2)(x)

x = BatchNormalization()(x)

x = Dense(4, activation = "relu")(x)

x = ActivityRegularization(l2 = 0.1)(x)

x = Dropout(0.2)(x)

x = Dense(1, activation = "sigmoid")(x)



model_LSTM = Model(inputs = inp, outputs = x)

model_LSTM.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



print(model_LSTM.summary())
checkpoint = ModelCheckpoint(

    'model_LSTM.h5', 

    monitor = 'val_loss', 

    verbose = 1, 

    save_best_only = True

)



reduce_lr = ReduceLROnPlateau(

    monitor = 'val_loss', 

    factor = 0.2, 

    verbose = 1, 

    patience = 5,                        

    min_lr = 0.001

)



model_LSTM.fit(x_train, y_train, batch_size = 32, epochs = 15, validation_data = (x_test, y_test), callbacks = [reduce_lr, checkpoint])
model_LSTM.load_weights('model_LSTM.h5')

pred_LSTM = model_LSTM.predict(x_test)



acc_LSTM = accuracy_score(y_test, np.where(pred_LSTM > 0.5, 1, 0))

f1_LSTM = f1_score(y_test, np.where(pred_LSTM > 0.5, 1, 0))



print(acc_LSTM, f1_LSTM)
inp = Input(shape = (MAX_WORDS, ))

x = Embedding(num_words, 300, weights = [embedding_matrix])(inp)

x = Conv1D(MAX_WORDS, 3)(x)

x = Activation('relu')(x)

x = MaxPooling1D(pool_size=2, strides=2)(x)

x = Bidirectional(LSTM(MAX_WORDS, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))(x)

x = ActivityRegularization(l2 = 0.1)(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.2)(x)

x = BatchNormalization()(x)

x = Dense(4, activation = "relu")(x)

x = ActivityRegularization(l2 = 0.1)(x)

x = Dropout(0.2)(x)

x = Dense(1, activation = "sigmoid")(x)



model_CNN_LSTM = Model(inputs = inp, outputs = x)

model_CNN_LSTM.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



print(model_CNN_LSTM.summary())
checkpoint = ModelCheckpoint(

    'model_CNN_LSTM.h5', 

    monitor = 'val_loss', 

    verbose = 1, 

    save_best_only = True

)



model_CNN_LSTM.fit(x_train, y_train, batch_size = 64, epochs = 10, validation_data = (x_test, y_test), callbacks = [reduce_lr, checkpoint])
model_CNN_LSTM.load_weights('model_CNN_LSTM.h5')

pred_CNN_LSTM = model_CNN_LSTM.predict(x_test)



acc_CNN_LSTM = accuracy_score(y_test, np.where(pred_CNN_LSTM > 0.5, 1, 0))

f1_CNN_LSTM = f1_score(y_test, np.where(pred_CNN_LSTM > 0.5, 1, 0))



print(acc_CNN_LSTM, f1_CNN_LSTM)
numerical_features = train_df[['tweet_sentiment', 'nr_of_char', 'nr_of_words', 'nr_of_unique_words', 'nr_of_punctuation', 'nr_of_stopwords']].to_numpy()
nlp_inp = Input(shape = (MAX_WORDS, ))

x = Embedding(num_words, 300, weights = [embedding_matrix])(nlp_inp)

x = Bidirectional(LSTM(MAX_WORDS,dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))(x)

x = ActivityRegularization(l2 = 0.1)(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.2)(x)

x = BatchNormalization()(x)

x = Dense(4, activation = "relu")(x)

x = ActivityRegularization(l2 = 0.1)(x)



num_features_inp = Input(shape = (6, ), name = 'num_features_inp')

y = Dense(4, activation = "relu")(num_features_inp)

x = ActivityRegularization(l2 = 0.1)(x)



z = Concatenate()([x, y])

z = Dropout(0.2)(z)

z = Dense(1, activation = "sigmoid")(z)



model_LSTM_FC = Model(inputs = [nlp_inp, num_features_inp], outputs = z)

model_LSTM_FC.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



print(model_LSTM_FC.summary())
x_train_text, x_test_text, x_train_num, x_test_num, y_train, y_test = train_test_split(train_pad, numerical_features, train_df['target'].values, test_size=0.25, random_state = 42)



print("Shape of train", x_train_text.shape)

print("Shape of Validation", x_test_text.shape)

print("Shape of train", x_train_num.shape)

print("Shape of Validation", x_test_num.shape)
checkpoint = ModelCheckpoint(

    'model_LSTM_FC.h5', 

    monitor = 'val_loss', 

    verbose = 1, 

    save_best_only = True

)



model_LSTM_FC.fit([x_train_text, x_train_num], y_train, batch_size=32, epochs=15, validation_data=([x_test_text, x_test_num], y_test), callbacks = [reduce_lr, checkpoint])
model_LSTM_FC.load_weights('model_LSTM_FC.h5')

pred_LSTM_FC = model_LSTM_FC.predict([x_test_text, x_test_num])



acc_LSTM_FC = accuracy_score(y_test, np.where(pred_LSTM_FC > 0.5, 1, 0))

f1_LSTM_FC = f1_score(y_test, np.where(pred_LSTM_FC > 0.5, 1, 0))



print(acc_LSTM_FC, f1_LSTM_FC)
print("Accuracy and F1-Score of the LSTM: ", acc_LSTM, f1_LSTM)

print("Accuracy and F1-Score of the CNN_LSTM: ", acc_CNN_LSTM, f1_CNN_LSTM)

print("Accuracy and F1-Score of the LSTM_FC: ", acc_LSTM_FC, f1_LSTM_FC)
!pip install transformers

!pip install simpletransformers
columns = train_df[["text_cleaned", "target"]]

train_df_V2 = columns.copy()
rename = {"text_cleaned": "text", "target": "labels"}

train_df_V2.rename(columns = rename, inplace=True)
train_x_y = train_df_V2.sample(frac = 0.75, random_state = 42)

test_x_y = pd.concat([train_df_V2, train_x_y]).drop_duplicates(keep=False)
from simpletransformers.classification import ClassificationModel, ClassificationArgs





model_args = ClassificationArgs()

model_args.use_early_stopping = True

model_args.early_stopping_delta = 0.01

model_args.early_stopping_metric = "mcc"

model_args.early_stopping_metric_minimize = False

model_args.early_stopping_patience = 5

model_args.evaluate_during_training_steps = 1000

model_args.reprocess_input_data = True

model_args.overwrite_output_dir = True

model_args.no_save = True



model_bert = ClassificationModel("bert", "bert-base-uncased", args=model_args, use_cuda=False)

model_bert.train_model(train_x_y)
pred_bert, out_bert = model_bert.predict(test_x_y['text'].values)



acc_bert = accuracy_score(test_x_y['labels'].to_numpy(), pred_bert)

f1_bert = f1_score(test_x_y['labels'].to_numpy(), pred_bert)



print(acc_bert, f1_bert)
model_bert = ClassificationModel("bert", "bert-base-uncased", args=model_args, use_cuda=False)

model_bert.train_model(train_df_V2)



final_pred_bert, final_out_bert = model_bert.predict(test_df['text_cleaned'].values)



submit = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

submit['target'] = final_pred_bert

submit.to_csv('submission.csv', index=False)