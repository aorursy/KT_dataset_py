import numpy as np

import pandas as pd

import plotly.express as px



import re

from nltk.stem import PorterStemmer

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight



import tensorflow as tf
data = pd.read_csv('../input/ramen-ratings/ramen-ratings.csv')
data
data.info()
data.isna().sum()
data['Top Ten'].unique()
data['Top Ten'] = data['Top Ten'].replace('\n', np.NaN)



data['isTopTen'] = data['Top Ten'].apply(lambda x: 0 if str(x) == 'nan' else 1)

data = data.drop('Top Ten', axis=1)



data = data.drop('Review #', axis=1)
data.isna().sum()
data = data.dropna(axis=0).reset_index(drop=True)
data.query("Stars == 'Unrated'")
data = data.drop(data.query("Stars == 'Unrated'").index, axis=0).reset_index(drop=True)
print("Total missing values:", data.isna().sum().sum())
data
ramen_names = data.loc[:, 'Variety']

ramen_names
ps = PorterStemmer()



def process_name(name):

    new_name = name.lower() # Make name lowercase

    new_name = re.sub(r'[^a-z0-9\s]', '', new_name) # Remove punctuation

    new_name = re.sub(r'[0-9]+', 'number', new_name) # Change numerical words to "number"

    new_name = new_name.split(" ") # Make string into a list of words

    new_name = list(map(lambda x: ps.stem(x), new_name)) # Stem each word

    new_name = list(map(lambda x: x.strip(), new_name)) # Removing leading and trailing whitespace

    for i in range(len(new_name)):

        if new_name[i] == 'flavour':

            new_name[i] = 'flavor'

    if '' in new_name:

        new_name.remove('') # Remove the empty string if it exists

    return new_name
ramen_names = ramen_names.apply(process_name)

ramen_names
# Getting the number of unique words in our list of ramen names

vocabulary = set()



for name in ramen_names:

    for word in name:

        if word not in vocabulary:

            vocabulary.add(word)



vocab_length = len(vocabulary)





# Getting the maximum length of a single ramen name

max_seq_length = max(ramen_names.apply(lambda x: len(x)))





# Print results

print("       Vocab length:", vocab_length)

print("Max sequence length:", max_seq_length)
tokenizer = Tokenizer(num_words=vocab_length)

tokenizer.fit_on_texts(ramen_names)



word_index = tokenizer.word_index



sequences = tokenizer.texts_to_sequences(ramen_names)





name_features = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
name_features
data = data.drop('Variety', axis=1)
data
def onehot_encode(df, columns, prefixes):

    df = df.copy()

    for column, prefix in zip(columns, prefixes):

        dummies = pd.get_dummies(df[column], prefix=prefix)

        df = pd.concat([df, dummies], axis=1)

        df = df.drop(column, axis=1)

    return df
data = onehot_encode(

    data,

    ['Brand', 'Style', 'Country'],

    ['B', 'S', 'C']

)
data
labels = data.loc[:, 'isTopTen']



other_features = data.drop('isTopTen', axis=1)
scaler = StandardScaler()



other_features = pd.DataFrame(scaler.fit_transform(other_features), columns=other_features.columns)
name_features_series = pd.Series(list(name_features), name='Name')
features = pd.concat([name_features_series, other_features], axis=1)

features
X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=34)
X_train_1 = np.stack(X_train['Name'].to_numpy())

X_train_2 = X_train.drop('Name', axis=1)



X_test_1 = np.stack(X_test['Name'].to_numpy())

X_test_2 = X_test.drop('Name', axis=1)
name_features.shape
other_features.shape
class_weights = dict(

    enumerate(

        class_weight.compute_class_weight(

            'balanced',

            y_train.unique(),

            y_train

        )

    )

)



class_weights
embedding_dim = 64



# Training on name features

name_inputs = tf.keras.Input(shape=(13,), name='name_inputs')



batch_norm = tf.keras.layers.BatchNormalization(name='batch_norm')(name_inputs)



name_embedding = tf.keras.layers.Embedding(

    input_dim=vocab_length,

    output_dim=embedding_dim,

    input_length=max_seq_length,

    name='name_embedding'

)(batch_norm)



name_outputs = tf.keras.layers.Flatten(name='name_flatten')(name_embedding)



# Training on other features

other_inputs = tf.keras.Input(shape=(401,), name='other_inputs')



hidden = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(other_inputs)

other_outputs = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(hidden)





# Concatenate outputs and make predictions

concat = tf.keras.layers.concatenate([name_outputs, other_outputs], name='concatenate')



outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(concat)





# Constructing and plotting model

model = tf.keras.Model(inputs=[name_inputs, other_inputs], outputs=outputs)



tf.keras.utils.plot_model(model)
model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[

        'accuracy',

        tf.keras.metrics.AUC(name='auc'),

        tf.keras.metrics.Precision(name='prec'),

        tf.keras.metrics.Recall(name='rec')

    ]

)





batch_size = 64

epochs = 100



history = model.fit(

    [X_train_1, X_train_2],

    y_train,

    validation_split=0.2,

    class_weight=class_weights,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau()

    ],

    verbose=0

)
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'x': "Epoch", 'y': "Loss"},

    title="Loss Over Time"

)



fig.show()
results = model.evaluate([X_test_1, X_test_2], y_test)



print(f"\n Accuracy: {results[1]:.5f}")

print(f"      AUC: {results[2]:.5f}")

print(f"Precision: {results[3]:.5f}")

print(f"   Recall: {results[4]:.5f}")
y_test.value_counts()
print(f"Percent of ramens that are Top 10: {y_test.mean() * 100:.1f}%")