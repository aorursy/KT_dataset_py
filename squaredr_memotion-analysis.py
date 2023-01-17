import tensorflow as tf

import numpy as np

import pandas as pd
data = pd.read_csv("../input/memotion-dataset-7k/memotion_dataset_7k/labels.csv")

data = data.iloc[:,3:]

data.head()
data = data.dropna()
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import string

import re



# Having looked at our data above, we see that the raw text contains HTML break

# tags of the form '<br />'. These tags will not be removed by the default

# standardizer (which doesn't strip HTML). Because of this, we will need to

# create a custom standardization function.

def custom_standardization(input_data):

    lowercase = tf.strings.lower(input_data)

    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")

    return tf.strings.regex_replace(

        stripped_html, "[%s]" % re.escape(string.punctuation), ""

    )





# Model constants.

max_features = 20000

embedding_dim = 128

sequence_length = 500



# Now that we have our custom standardization, we can instantiate our text

# vectorization layer. We are using this layer to normalize, split, and map

# strings to integers, so we set our 'output_mode' to 'int'.

# Note that we're using the default split function,

# and the custom standardization defined above.

# We also set an explicit maximum sequence length, since the CNNs later in our

# model won't support ragged sequences.

vectorize_layer = TextVectorization(

    standardize=custom_standardization,

    max_tokens=max_features,

    output_mode="int",

    output_sequence_length=sequence_length,

)



# Now that the vocab layer has been created, call `adapt` on a text-only

# dataset to create the vocabulary. You don't have to batch, but for very large

# datasets this means you're not keeping spare copies of the dataset in memory.



# Let's make a text-only dataset (no labels):

text_ds = np.asarray(data["text_corrected"].values)

# Let's call `adapt`:

vectorize_layer.adapt(text_ds)
Y = pd.get_dummies(data.iloc[:,1:], columns = ['sarcasm', 'offensive', 'motivational', 'overall_sentiment', 'humour'])

Y.head()
Y.shape
from tensorflow.keras import layers



inputs = tf.keras.Input(shape=(None,), dtype="int64")



X = layers.Embedding(max_features, embedding_dim)(inputs)

X = layers.Dropout(0.5)(X)



X = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(X)

X = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(X)

X = layers.GlobalMaxPooling1D()(X)



X = layers.Dense(128, activation="relu")(X)

X = layers.Dropout(0.5)(X)



predictions = layers.Dense(Y.shape[1], activation='softmax', name='Predictions')(X)



model = tf.keras.Model(inputs, predictions)



model.compile(loss = 'categorocal_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()