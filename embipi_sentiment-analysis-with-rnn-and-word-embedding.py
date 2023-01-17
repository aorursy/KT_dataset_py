## Importing libraries

import pandas as pd

pd.set_option('display.max_colwidth', -1)



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences



from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin



from keras import optimizers

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, GRU, BatchNormalization, Dropout

from keras.initializers import Constant



import gensim

import os

import re 

import string



from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



# import nltk

# nltk.download("stopwords")

# nltk.download('punkt')
## Reading the data

data_train = pd.read_csv("../input/sentiment-data-train/sentiment_train_data.csv", sep=",", header=None, encoding="latin1")

data_test = pd.read_csv("../input/sentiment-data/sentiment_test_data.csv", sep=",", header=None, encoding="latin1")

                      

## Assigning names to the columns

data_train.columns = ["Polarity", "ID", "Date", "Query", "User", "Text"]

data_test.columns = ["Polarity", "ID", "Date", "Query", "User", "Text"]



## Joining train and test data sets

all_data = pd.concat([data_train, data_test]).reset_index(drop=True)



## Eliminating the "Neutral" category of the target

all_data = all_data[all_data.Polarity != 2]



## Changing the target to 0 and 1 values

all_data.Polarity = all_data.Polarity.apply(lambda x: 0 if x==0 else 1)



## Viewing a sample and the shape of the final set

all_data.sample(5)

print("El tamaño del data set es:", all_data.shape)
## Changing the index order

all_data = all_data.reindex(np.random.permutation(all_data.index))  

all_data = all_data[["Polarity", "Text"]].reset_index(drop=True)



## Obtaining a small sample of the original set by a ramdon subsampling

data = all_data.sample(int(len(all_data)*.2))



## Checking the target balance for both positive and negative categories

data.groupby("Polarity").count()
## Creating a object class to perform the data cleaning



class CleanText(BaseEstimator, TransformerMixin):

    def remove_mentions(self, input_text):

        return re.sub(r'@\w+', '', input_text)

    

    def remove_urls(self, input_text):

        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

    

    def emoji_oneword(self, input_text):

        # By compressing the underscore, the emoji is kept as one word

        return input_text.replace('_','')

    

    def remove_punctuation(self, input_text):

        # Make translation table

        punct = string.punctuation

        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space

        return input_text.translate(trantab)



    def remove_digits(self, input_text):

        return re.sub('\d+', '', input_text)

    

    def to_lower(self, input_text):

        return input_text.lower()

    

    def remove_stopwords(self, input_text):

        stopwords_list = stopwords.words('english')

        # Some words which might indicate a certain sentiment are kept via a whitelist

        whitelist = ["n't", "not", "no"]

        words = input_text.split() 

        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 

        return " ".join(clean_words) 

       

    def fit(self, X, y=None, **fit_params):

        return self

    

    def transform(self, X, **transform_params):

        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords)

        return clean_X
## Generating a CleanText objetc to perform the data cleaning

ct = CleanText()

data_clean = ct.fit_transform(data.Text)



## Checking for empty tweets after the cleaning and filling them

empty_clean = data_clean == ""

print("{} tweets vacios tras la limpieza de datos".format(data_clean[empty_clean].count()))

data_clean.loc[empty_clean] = '[no_text]'



## Overwritting the tweets

data["Text"] = data_clean

data.sample(5)
## Obtening the train and test samples

X_train, X_test, y_train, y_test = train_test_split(data.drop("Polarity", axis=1),

                                                    data.Polarity,

                                                    test_size=0.2,

                                                    random_state=888)



## Creating vectors with the texts and targets

X_train = X_train.Text.values

y_train = y_train.values



X_test = X_test.Text.values

y_test= y_test.values



## Viewing the shape of each data setbservamos las dimensiones de cada set de datos

print("Tamaño del set de entrenamiento:", X_train.shape)

print("Tamaño del target de entrenamiento:", y_train.shape)



print("\nTamaño del set de testeo:", X_test.shape)

print("Tamaño del target de testeo:", y_test.shape)
## Joining the data on a unique set

total_tweets = np.concatenate((X_train, X_test), axis=0)



## Generating a Tokenizer object to vectorize the tweets obtaining a index for each word

token = Tokenizer()

token.fit_on_texts(total_tweets)



## Checking the max length of the tweets

max_length = max([len(word.split()) for word in total_tweets])

print("La longitud máxima de los tweets es", max_length)



## Calculating the dimension of the word frequencies dictionary

vocabulary_dim = len(token.word_index) + 1

print("El tamaño del diccionario es", vocabulary_dim)



## Replacing the words of each tweet by their index

X_train_token = token.texts_to_sequences(X_train)

X_test_token = token.texts_to_sequences(X_test)



## Transforming the preview list into a matrix

X_train = pad_sequences(X_train_token, maxlen=max_length, padding="post")

X_test = pad_sequences(X_test_token, maxlen=max_length, padding="post")
## Generating a sequential model

model = Sequential()



## Adding a embedding matrix layer

model.add(Embedding(input_dim=vocabulary_dim,  output_dim=100, input_length=max_length))



## Adding a GRU module layer

model.add(GRU(units=64, dropout=0.5, recurrent_dropout=0.5))



## Adding a full-conected layer

model.add(Dense(32, activation="relu"))



## Adding a normalization layer 

model.add(BatchNormalization())



## Adding a dropout layer

model.add(Dropout(0.4))



## Adding a full-conected layer

model.add(Dense(16, activation="relu"))



## Adding a normalization layer 

model.add(BatchNormalization())



## Adding a full-conected layer with a sigmoid activation function  

model.add(Dense(1, activation="sigmoid"))



## Compiling the cost function, metrics and optimizer

model.compile(optimizer="adam",

              loss="binary_crossentropy",          

              metrics=["accuracy"])



## Showing a summary of the neural network structure

model.summary()
## Fitting the model

model_history = model.fit(x=X_train,

                          y=y_train,

                          batch_size=512,

                          epochs=20,

                          validation_split=0.2,

                          verbose=1)
## Creating a dataframe to be able to generate a visualization

history = pd.DataFrame(model_history.history)



## Including the epoch for each error

history['epoch'] = model_history.epoch



# Joining the errors on one column to generate a visulization with seaborn

df = history.melt(id_vars='epoch',

                  var_name='Type',

                  value_name='Accuracy',

                  value_vars=['acc','val_acc'])



# Generating the graph

fig, ax = plt.subplots(figsize=(16,8))

_ = sns.lineplot(x='epoch', y='Accuracy', hue='Type', data=df)



print("Accuracy del entrenamiento: %.2f" % model_history.model.evaluate(X_train, y_train, verbose=0)[1]) 

print("Accuracy de la validación: %.2f" % model_history.model.evaluate(X_test, y_test, verbose=0)[1])
## Generating a sequential model

model = Sequential()



## Adding a embedding matrix layer

model.add(Embedding(input_dim=vocabulary_dim,  output_dim=100, input_length=max_length))



## Adding a LSTM module layer

model.add(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5))



## Adding a full-conected layer

model.add(Dense(32, activation="relu"))



## Adding a normalization layer 

model.add(BatchNormalization())



## Adding a dropout layer

model.add(Dropout(0.4))



## Adding a full-conected layer

model.add(Dense(16, activation="relu"))



## Adding a normalization layer 

model.add(BatchNormalization())



## Adding a full-conected layer with a sigmoid activation function  

model.add(Dense(1, activation="sigmoid"))



## Compiling the cost function, metrics and optimizer

model.compile(optimizer="adam",

              loss="binary_crossentropy",          

              metrics=["accuracy"])



## Showing a summary of the neural network structure

model.summary()
## Fitting the model

model_history = model.fit(x=X_train,

                          y=y_train,

                          batch_size=512,

                          epochs=20,

                          validation_split=0.2,

                          verbose=1)
## Creating a dataframe to be able to generate a visualization

history = pd.DataFrame(model_history.history)



## Including the epoch for each error

history['epoch'] = model_history.epoch



# Joining the errors on one column to generate a visulization with seaborn

df = history.melt(id_vars='epoch',

                  var_name='Type',

                  value_name='Accuracy',

                  value_vars=['acc','val_acc'])



# Generating the graph

fig, ax = plt.subplots(figsize=(16,8))

_ = sns.lineplot(x='epoch', y='Accuracy', hue='Type', data=df)



print("Accuracy del entrenamiento: %.2f" % model_history.model.evaluate(X_train, y_train, verbose=0)[1]) 

print("Accuracy de la validación: %.2f" % model_history.model.evaluate(X_test, y_test, verbose=0)[1])
## Generating a empty list and changing the format of the total set

review_lines = list()

lines = total_tweets.tolist()



## Generating a loop to clean the words

for i in lines:   

    ## Tokenizing the words of each tweet

    tokens = word_tokenize(i)

    

    ## Transforming the tokens to lower case

    tokens = [w.lower() for w in tokens]

    

    ## Eliminating punctuations of each word

    table = str.maketrans("", "", string.punctuation)

    stripped = [w.translate(table) for w in tokens]

    

    ## Eliminating the numeric tokens

    words = [word for word in stripped if word.isalpha()]

    review_lines.append(words)

    

print("Número de tweets tratados en el corpus de texto:", len(review_lines))
## Training the Word2Vec model

model = gensim.models.Word2Vec(sentences=review_lines,

                               size=100,

                               window=5,

                               workers=5,

                               min_count=1)



print("Tamaño del vocabulario:", len(list(model.wv.vocab)))
## Similarity test of words from the model

model.wv.most_similar("good")
## Out-of-context test to find odd words

model.wv.doesnt_match("mouse cat dog pizza frog".split())
## Saving the result of the model 

model.wv.save_word2vec_format("../input/sentiment-data/tweets_embedding_word2vec.txt", binary=False)



## Loading the result of the model

file = open(os.path.join("", "tweets_embedding_word2vec.txt"), encoding="utf-8")



## Generating a empty dictionary and filling it with words and their coefficients

embeddings_index={}

for lines in file:

    values = lines.split()

    word = values[0]

    coefs = np.asarray(values[1:])

    embeddings_index[word] = coefs

file.close()
## Generating a Tokenizer object to vectorize the text samples indexing the words

token_W2V = Tokenizer()

token_W2V.fit_on_texts(review_lines)



## Calculating the number of unique tokens

word_index = token_W2V.word_index

print("Tokens únicos encontrados:", len(word_index))



## Replacing the words of each tweets by their index

sequences = token_W2V.texts_to_sequences(review_lines)



## Transforming the preview list into a matrix recovering the target of both sets

review_matrix = pad_sequences(sequences, maxlen=max_length, padding="post")

print("Tamaño del tensor review:", review_matrix.shape)



sentiment = np.concatenate((y_train, y_test), axis=0)

print("Tamaño del tensor sentiment", sentiment.shape)
## Generating a matrix of zeros

num_words = len(word_index) + 1

print("El número de palabras es:", num_words)



embedding_matrix = np.zeros((num_words, 100))



## Filling the matrix with the generated embedding from Word2Vec

for word, i in word_index.items():

    if i > num_words:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
## Generating a ramdon subsampling to obtaint the validation set

ind = np.arange(review_matrix.shape[0])

np.random.shuffle(ind)



review_matrix = review_matrix[ind]

sentiment = sentiment[ind]



num_validation_samples = int(0.2 * review_matrix.shape[0])



X_train_W2V = review_matrix[:-num_validation_samples]

y_train_W2V = sentiment[:-num_validation_samples]

X_test_W2V = review_matrix[-num_validation_samples:]

y_test_W2V = sentiment[-num_validation_samples:]



## Viewing the shape of each data set

print("Tamaño del set de entrenamiento:", X_train_W2V.shape)

print("Tamaño del target de entrenamiento:", y_train_W2V.shape)



print("\nTamaño del set de testeo:", X_test_W2V.shape)

print("Tamaño del target de testeo:", y_test_W2V.shape)
## Generating a sequential model

model = Sequential()



## Adding a embeddint matrix layer

model.add(Embedding(input_dim=num_words,

                    output_dim=100,

                    embeddings_initializer=Constant(embedding_matrix),

                    input_length=max_length,

                    trainable=False))



## Adding a GRU module layer

model.add(GRU(units=64, dropout=0.5, recurrent_dropout=0.5))



## Adding a full-conected layer

model.add(Dense(32, activation="relu"))



## Adding a normalization layer 

model.add(BatchNormalization())



## Adding a dropout layer

model.add(Dropout(0.4))



## Adding a full-conected layer

model.add(Dense(16, activation="relu"))



## Adding a normalization layer 

model.add(BatchNormalization())



## Adding a full-conected layer with a sigmoid activation function 

model.add(Dense(1, activation="sigmoid"))



## Compiling the cost function, the metrics and the optimizer

model.compile(optimizer="adam",

              loss="binary_crossentropy",          

              metrics=["accuracy"])



## Showing a summary of the neural network structure

model.summary()
## Fitting the model

model_history = model.fit(x=X_train_W2V,

                          y=y_train_W2V,

                          batch_size=512,

                          epochs=20,

                          validation_split=0.2,

                          verbose=1)
## Creating a dataframe to be able to generate a visualization

history = pd.DataFrame(model_history.history)



## Including the epoch for each error

history['epoch'] = model_history.epoch



# Joining the errors on one column to generate a visulization with seaborn

df = history.melt(id_vars='epoch',

                  var_name='Type',

                  value_name='Accuracy',

                  value_vars=['acc','val_acc'])



# Generating the graph

fig, ax = plt.subplots(figsize=(16,8))

_ = sns.lineplot(x='epoch', y='Accuracy', hue='Type', data=df)



print("Accuracy del entrenamiento: %.2f" % model_history.model.evaluate(X_train_W2V, y_train_W2V, verbose=0)[1]) 

print("Accuracy de la validación: %.2f" % model_history.model.evaluate(X_test_W2V, y_test_W2V, verbose=0)[1])
## Generating a sequential model

model = Sequential()



## Adding a embeddint matrix layer

model.add(Embedding(input_dim=num_words,

                    output_dim=100,

                    embeddings_initializer=Constant(embedding_matrix),

                    input_length=max_length,

                    trainable=False))



## Adding a GRU module layer

model.add(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5))



## Adding a full-conected layer

model.add(Dense(32, activation="relu"))



## Adding a normalization layer 

model.add(BatchNormalization())



## Adding a dropout layer

model.add(Dropout(0.4))



## Adding a full-conected layer

model.add(Dense(16, activation="relu"))



## Adding a normalization layer 

model.add(BatchNormalization())



## Adding a full-conected layer with a sigmoid activation function 

model.add(Dense(1, activation="sigmoid"))



## Compiling the cost function, the metrics and the optimizer

model.compile(optimizer="adam",

              loss="binary_crossentropy",          

              metrics=["accuracy"])



## Showing a summary of the neural network structure

model.summary()
## Fitting the model

model_history = model.fit(x=X_train_W2V,

                          y=y_train_W2V,

                          batch_size=512,

                          epochs=20,

                          validation_split=0.2,

                          verbose=1)
## Creating a dataframe to be able to generate a visualization

history = pd.DataFrame(model_history.history)



## Including the epoch for each error

history['epoch'] = model_history.epoch



# Joining the errors on one column to generate a visulization with seaborn

df = history.melt(id_vars='epoch',

                  var_name='Type',

                  value_name='Accuracy',

                  value_vars=['acc','val_acc'])



# Generating the graph

fig, ax = plt.subplots(figsize=(16,8))

_ = sns.lineplot(x='epoch', y='Accuracy', hue='Type', data=df)



print("Accuracy del entrenamiento: %.2f" % model_history.model.evaluate(X_train_W2V, y_train_W2V, verbose=0)[1]) 

print("Accuracy de la validación: %.2f" % model_history.model.evaluate(X_test_W2V, y_test_W2V, verbose=0)[1])