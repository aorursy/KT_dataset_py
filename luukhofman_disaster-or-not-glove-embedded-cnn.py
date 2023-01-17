! pip install pyspellchecker
import string

import nltk

import keras



import numpy as np 

import pandas as pd 

import tensorflow as tf

import regex as re



from nltk.corpus import stopwords

from spellchecker import SpellChecker
# Data import

train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



tweets = [tweet for tweet in train_data['text']]

targets = [target for target in train_data['target']]

test_tweets = [tweet for tweet in test_data['text']]
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(text):

    

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    

    return emoji_pattern.sub(r'', text)



tweets = [remove_emoji(tweet) for tweet in tweets]
def remove_punct(text):

    

    table=str.maketrans('','',string.punctuation)

    

    return text.translate(table)



tweets = [remove_punct(tweet) for tweet in tweets]
# Function based on -> https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings

def clean_text(text):

    

    # Convert words to lower case and split them

    text = text.lower().split()

    

    # Remove stop words

    stops = set(stopwords.words("english"))

    text = [w for w in text if not w in stops and len(w) >= 3]

    

    text = " ".join(text)

    

    # Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

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

    

    return text



# Cleaning tweets

tweets = [clean_text(tweet) for tweet in tweets]
def remove_URL(text):

    

    result = re.sub(r"http\S+", "", text)

    

    return result



tweets = [remove_URL(tweet) for tweet in tweets]
# Function that loads the glove embedding into a dictionary and assigns an embedding to each word 

def embed_words(file_link):

    

    embedded_words = {}



    with open(file_link) as f:

        for line in f:

            values = line.split()

            word = values[0]

            embedding = np.asarray(values[1:101])

            embedded_words[word] = embedding

            

        return embedded_words

 

# Function call -> get word embeddings

embedded_words = embed_words('/kaggle/input/glove-embedding/glove.6B.100d.txt')
# Initiate tokenizer & create word tokens 

tokenizer = keras.preprocessing.text.Tokenizer()

tokenizer.fit_on_texts(tweets)

num_of_words = len(tokenizer.word_index) + 1
# Function that maps each token_ID of word in training data to a embedding vector from the glove embedding 

def create_embedding_matrix(vocab_size, tokenizer):

    

    embedding_matrix = np.zeros((vocab_size, 100))

    

    for word, i in tokenizer.word_index.items():

        

        try:            

            embedding = embedded_words[word]

            embedding_matrix[i] = embedding

        except:

            

            pass

        

    return embedding_matrix



# Apply function

embedding_matrix = create_embedding_matrix(num_of_words, tokenizer)
# Sequence encode

encoded_docs = tokenizer.texts_to_sequences(tweets)

max_length = max([len(tweet) for tweet in tweets])



# Sad sequences

x_train = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# Initiate a  Nmodel 

model = keras.models.Sequential()



# The embedding layer which maps the vocab indices into embedding_dims dimensions

model.add(keras.layers.Embedding(num_of_words,100, input_length=max_length))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv1D(3,3, padding='valid',activation='relu', strides=1))

model.add(keras.layers.GlobalMaxPooling1D())

model.add(keras.layers.Dense(20))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dense(1))

model.add(keras.layers.Activation('sigmoid'))



# Get model summary

model.summary()



# Show if GPU is avialable

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
model.compile(optimizer = 'adam',

              loss ='binary_crossentropy',

              metrics = ['accuracy'])



model.fit(x = x_train, y = targets,

          batch_size = 32, epochs = 10, 

          verbose = 1, validation_split = 0.1,

          shuffle = True, workers = 6, 

          use_multiprocessing = True)
# Make predictions

test_encoded_docs = tokenizer.texts_to_sequences(test_tweets)

x_test = keras.preprocessing.sequence.pad_sequences(test_encoded_docs, maxlen=max_length, padding='post')

predictions = model.predict_classes(x_test)



# Write submission

submission = {"id":None, "target":None}



submission["id"] = [id for id in test_data['id']]

submission["target"] = [prediction for prediction in predictions]



pd.DataFrame.from_dict(data=submission).to_csv('disaster_submission.csv', header=True, index=False)