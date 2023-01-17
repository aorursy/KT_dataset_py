# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns



df_train= pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

# Input data files are available in the "../input/" directory.



# Any results you write to the current directory are saved as output.



def clean_text(text):

    import re

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)

    text = re.sub(r"you'll", "you will", text)

    text = re.sub(r"i'll", "i will", text)

    text = re.sub(r"she'll", "she will", text)

    text = re.sub(r"he'll", "he will", text)

    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"she's", "she is", text)

    text = re.sub(r"that's", "that is", text)

    text = re.sub(r"what's", "what is", text)

    text = re.sub(r"where's", "where is", text)

    text = re.sub(r"there's", "there is", text)

    text = re.sub(r"here's", "here is", text)

    text = re.sub(r"who's", "who is", text)

    text = re.sub(r"how's", "how is", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"can't", "cannot", text)

    text = re.sub(r"won't", "will not", text)

    text = re.sub(r"don't", "do not", text)

    text = re.sub(r"shouldn't", "should not", text)

    text = re.sub(r"n't", " not", text)

    text = re.sub(r"   ", " ", text) # Remove any extra spaces

    return text

def massage_text(text):

    import re

    from nltk.corpus import stopwords

    ## remove anything other then characters and put everything in lowercase

    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text)

    tweet = re.sub("[^a-zA-Z]", ' ', text)

    tweet = tweet.lower()

    tweet = tweet.split()



    from nltk.stem import WordNetLemmatizer

    lem = WordNetLemmatizer()

    tweet = [lem.lemmatize(word) for word in tweet

             if word not in set(stopwords.words('english'))]

    tweet = ' '.join(tweet)

    return tweet



def plot_history(history):

    import matplotlib.pyplot as plt

    acc = history.history['accuracy']

    loss = history.history['loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.title('Training and validation loss')

    plt.legend()

    

def create_embedding_matrix(filepath, word_index, embedding_dim):

    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index

    embedding_matrix = np.zeros((vocab_size, embedding_dim))



    with open(filepath) as f:

        for line in f:

            word, *vector = line.split()

            if word in word_index:

                idx = word_index[word] 

                embedding_matrix[idx] = np.array(

                    vector, dtype=np.float32)[:embedding_dim]



    return embedding_matrix



df_train['clean_text'] = df_train['text'].apply(clean_text)

df_test['clean_text'] = df_test['text'].apply(clean_text)



df_train['clean_text'] = df_train['clean_text'].apply(massage_text)

df_test['clean_text'] = df_test['clean_text'].apply(massage_text)

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras import layers

from sklearn.model_selection import train_test_split



sentences_train, sentences_test, target_train, target_test = train_test_split(df_train['clean_text'], df_train['target'], test_size=0.01, random_state=42)

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)

X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1

 ## 0 index is reserved with no words in it

maxlen = 200



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

## always pad to make sure all sequences are of the same length

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.layers.normalization import BatchNormalization



## using create_model gave acc of 80

embedding_dim = 50

def create_model(embedding_dim,vocab_size,maxlen):

    embedding_matrix = create_embedding_matrix('/kaggle/input/embeded/glove.6B.50d.txt',tokenizer.word_index, embedding_dim)

    model = Sequential()

    model.add(layers.Embedding(input_dim=vocab_size, 

                               output_dim=embedding_dim, 

                               weights=[embedding_matrix],

                               input_length=maxlen,

                              trainable=False))

    model.add(layers.Conv1D(128, 5, activation='relu'))

    model.add(layers.GlobalMaxPool1D())

    model.add(layers.Dense(20, activation='relu'))

    model.add(BatchNormalization())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(20, activation='relu'))

    model.add(BatchNormalization())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    return model



batch_size = [10, 20, 40, 60, 80, 100]

epochs = [10, 50, 100]

model = KerasClassifier(build_fn=create_model,embedding_dim=embedding_dim,vocab_size=vocab_size, maxlen=maxlen,verbose=1) ##verbose = 1 will show the run for each epoch

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3) ##By default, the grid search will only use one thread. By setting the n_jobs argument in the GridSearchCV constructor to -1, the process will use all cores on your machine

grid_result = grid.fit(X_train, target_train)

grid_result.best_params_ #epcoh=10 batch_size = 60 this might be due to overfitting, because test acc dropped to 0.78

from sklearn.model_selection import cross_val_score

model = create_model(embedding_dim,vocab_size,maxlen)

model.fit(X_train,target_train,epochs=10,

          verbose=1,

          batch_size=60,         

          )



loss,accuracy = model.evaluate(X_train, target_train, verbose=1)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, target_test, verbose=1)

print("Testing Accuracy:  {:.4f}".format(accuracy))

test = df_test['clean_text']

sample_sub=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

X_tester = tokenizer.texts_to_sequences(test)

X_tester = pad_sequences(X_tester, padding='post', maxlen=maxlen)

y_pre=model.predict(X_tester)

submit = []

for x in y_pre:

    if(x<0.6):

        submit.append(0)

    else:

        submit.append(1)

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':submit})

sub.to_csv('submission.csv',index=False)
## trying LSTM model

## results: LSTM model is not accurate.

embedding_matrix = create_embedding_matrix('/kaggle/input/embeded200/glove.twitter.27B.200d.txt',tokenizer.word_index, embedding_dim)

embedding_matrix.shape

from keras.layers import LSTM

model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, 

                               output_dim=50, 

                               weights=[embedding_matrix],

                               input_length=maxlen,

                              trainable=True))

model.add(layers.Conv1D(128, 5, activation='relu'))

model.add(layers.MaxPooling1D(pool_size=5, strides=1))

model.add(layers.Bidirectional(LSTM(20, return_sequences=True), input_shape=(3, 1)))

model.add(layers.Bidirectional(LSTM(20, return_sequences=True), input_shape=(3, 1)))

model.add(layers.Bidirectional(LSTM(20, return_sequences=False), input_shape=(3, 1)))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, target_train, validation_data=(X_test, target_test), epochs=10, batch_size=10)



test = df_test['clean_text']

sample_sub=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

X_tester = tokenizer.texts_to_sequences(test)

X_tester = pad_sequences(X_tester, padding='post', maxlen=maxlen)

y_pre=model.predict(X_tester)

submit = []

for x in y_pre:

    if(x<0.5):

        submit.append(0)

    else:

        submit.append(1)

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':submit})

sub.to_csv('submission.csv',index=False)
## try training my own word embedding

from gensim.models import Word2Vec

def text_preprocessing(text):

    '''

    input: string to be processed

    output: preprocssed string

    '''

    text = text.lower() # make everything lower case

    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text) #remove url

    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE)) #remove punctuation

    text = " ".join([word for word in str(text).split() if word not in STOPWORDS]) #remove stop words

    text = " ".join([stemmer.stem(word) for word in text.split()])

    

    return text