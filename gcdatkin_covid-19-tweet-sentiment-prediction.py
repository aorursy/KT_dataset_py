import numpy as np

import pandas as pd

import plotly.express as px



import re

from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



import tensorflow as tf
train_df = pd.read_csv('../input/covid-19-nlp-text-classification/Corona_NLP_train.csv', encoding='latin-1')

test_df = pd.read_csv('../input/covid-19-nlp-text-classification/Corona_NLP_test.csv', encoding='latin-1')
train_df
test_df
train_df.info()
test_df.info()
train_inputs = train_df['OriginalTweet'].copy()

test_inputs = test_df['OriginalTweet'].copy()



train_labels = train_df['Sentiment'].copy()

test_labels = test_df['Sentiment'].copy()
sentiment_encoding = {

    'Extremely Negative': 0,

    'Negative': 0,

    'Neutral': 1,

    'Positive': 2,

    'Extremely Positive': 2

}



train_labels = train_labels.replace(sentiment_encoding)

test_labels = test_labels.replace(sentiment_encoding)
train_inputs
stop_words = stopwords.words('english')



def process_tweet(tweet):

    

    # remove urls

    tweet = re.sub(r'http\S+', ' ', tweet)

    

    # remove html tags

    tweet = re.sub(r'<.*?>', ' ', tweet)

    

    # remove digits

    tweet = re.sub(r'\d+', ' ', tweet)

    

    # remove hashtags

    tweet = re.sub(r'#\w+', ' ', tweet)

    

    # remove mentions

    tweet = re.sub(r'@\w+', ' ', tweet)

    

    #removing stop words

    tweet = tweet.split()

    tweet = " ".join([word for word in tweet if not word in stop_words])

    

    return tweet



# Function taken from @Shahraiz's wonderful notebook
train_inputs = train_inputs.apply(process_tweet)

test_inputs = test_inputs.apply(process_tweet)
train_inputs
max_seq_length = np.max(train_inputs.apply(lambda tweet: len(tweet)))
tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_inputs)



vocab_length = len(tokenizer.word_index) + 1





train_inputs = tokenizer.texts_to_sequences(train_inputs)

test_inputs = tokenizer.texts_to_sequences(test_inputs)



train_inputs = pad_sequences(train_inputs, maxlen=max_seq_length, padding='post')

test_inputs = pad_sequences(test_inputs, maxlen=max_seq_length, padding='post')
print("Vocab length:", vocab_length)

print("Max sequence length:", max_seq_length)
train_inputs.shape
embedding_dim = 16





inputs = tf.keras.Input(shape=(max_seq_length,), name='input_layer')



embedding = tf.keras.layers.Embedding(

    input_dim=vocab_length,

    output_dim=embedding_dim,

    input_length=max_seq_length,

    name='word_embedding'

)(inputs)



gru_layer = tf.keras.layers.Bidirectional(

    tf.keras.layers.GRU(units=256, return_sequences=True, name='gru_layer'),

    name='bidirectional_layer'

)(embedding)



max_pooling = tf.keras.layers.GlobalMaxPool1D(name='max_pooling')(gru_layer)



dropout_1 = tf.keras.layers.Dropout(0.4, name='dropout_1')(max_pooling)



dense = tf.keras.layers.Dense(64, activation='relu', name='dense')(dropout_1)



dropout_2 = tf.keras.layers.Dropout(0.4, name='dropout_2')(dense)



outputs = tf.keras.layers.Dense(3, activation='softmax', name='output_layer')(dropout_2)





model = tf.keras.Model(inputs=inputs, outputs=outputs)



print(model.summary())



tf.keras.utils.plot_model(model)
model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)





batch_size = 32

epochs = 2



history = model.fit(

    train_inputs,

    train_labels,

    validation_split=0.12,

    batch_size=batch_size,

    epochs=epochs,

    verbose=2

)
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'index': "epoch", 'value': "loss"}

)



fig.show()
model.evaluate(test_inputs, test_labels)