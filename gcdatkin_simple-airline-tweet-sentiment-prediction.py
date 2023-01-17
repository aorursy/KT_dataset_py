import numpy as np

import pandas as pd



import re

import emoji

from nltk.stem import PorterStemmer

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')
data
confidence_threshold = 0.6



data = data.drop(data.query("airline_sentiment_confidence < @confidence_threshold").index, axis=0).reset_index(drop=True)
tweets_df = pd.concat([data['text'], data['airline_sentiment']], axis=1)

tweets_df
tweets_df.isna().sum().sum()
tweets_df['airline_sentiment'].value_counts()
sentiment_ordering = ['negative', 'neutral', 'positive']



tweets_df['airline_sentiment'] = tweets_df['airline_sentiment'].apply(lambda x: sentiment_ordering.index(x))
tweets_df
emoji.demojize('@AmericanAir right on cue with the delays👌')
ps = PorterStemmer()



def process_tweet(tweet):

    new_tweet = tweet.lower()

    new_tweet = re.sub(r'@\w+', '', new_tweet) # Remove @s

    new_tweet = re.sub(r'#', '', new_tweet) # Remove hashtags

    new_tweet = re.sub(r':', ' ', emoji.demojize(new_tweet)) # Turn emojis into words

    new_tweet = re.sub(r'http\S+', '',new_tweet) # Remove URLs

    new_tweet = re.sub(r'\$\S+', 'dollar', new_tweet) # Change dollar amounts to dollar

    new_tweet = re.sub(r'[^a-z0-9\s]', '', new_tweet) # Remove punctuation

    new_tweet = re.sub(r'[0-9]+', 'number', new_tweet) # Change number values to number

    new_tweet = new_tweet.split(" ")

    new_tweet = list(map(lambda x: ps.stem(x), new_tweet)) # Stemming the words

    new_tweet = list(map(lambda x: x.strip(), new_tweet)) # Stripping whitespace from the words

    if '' in new_tweet:

        new_tweet.remove('')

    return new_tweet
tweets = tweets_df['text'].apply(process_tweet)



labels = np.array(tweets_df['airline_sentiment'])
tweets
# Get size of vocabulary

vocabulary = set()



for tweet in tweets:

    for word in tweet:

        if word not in vocabulary:

            vocabulary.add(word)



vocab_length = len(vocabulary)



# Get max length of a sequence

max_seq_length = 0



for tweet in tweets:

    if len(tweet) > max_seq_length:

        max_seq_length = len(tweet)



# Print results

print("Vocab length:", vocab_length)

print("Max sequence length:", max_seq_length)
tokenizer = Tokenizer(num_words=vocab_length)

tokenizer.fit_on_texts(tweets)



sequences = tokenizer.texts_to_sequences(tweets)



word_index = tokenizer.word_index



model_inputs = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
model_inputs
model_inputs.shape
X_train, X_test, y_train, y_test = train_test_split(model_inputs, labels, train_size=0.7, random_state=22)
embedding_dim = 32





inputs = tf.keras.Input(shape=(max_seq_length,))



embedding = tf.keras.layers.Embedding(

    input_dim=vocab_length,

    output_dim=embedding_dim,

    input_length=max_seq_length

)(inputs)





# Model A (just a Flatten layer)

flatten = tf.keras.layers.Flatten()(embedding)



# Model B (GRU with a Flatten layer)

gru = tf.keras.layers.GRU(units=embedding_dim)(embedding)

gru_flatten = tf.keras.layers.Flatten()(gru)



# Both A and B are fed into the output

concat = tf.keras.layers.concatenate([flatten, gru_flatten])



outputs = tf.keras.layers.Dense(3, activation='softmax')(concat)





model = tf.keras.Model(inputs, outputs)



tf.keras.utils.plot_model(model)
model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)





batch_size = 32

epochs = 100



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[

        tf.keras.callbacks.EarlyStopping(

            monitor='val_loss',

            patience=3,

            restore_best_weights=True,

            verbose=1

        ),

        tf.keras.callbacks.ReduceLROnPlateau()

    ]

)
model.evaluate(X_test, y_test)