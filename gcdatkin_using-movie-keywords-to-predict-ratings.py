import numpy as np

import pandas as pd

import plotly.express as px

from tqdm import tqdm



from ast import literal_eval

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split



import tensorflow as tf

import tensorflow_addons as tfa



from sklearn.decomposition import PCA
ratings_df = pd.read_csv('../input/the-movies-dataset/ratings.csv')

keywords_df = pd.read_csv('../input/the-movies-dataset/keywords.csv')
ratings_df
keywords_df
keywords_df.columns = ['movieId', 'keywords']
keywords_df.loc[0, 'keywords']
word_dictionary = {}



for word_list in tqdm(keywords_df['keywords']):

    for word in literal_eval(word_list):

        word_dictionary[word['id']] = word['name']
len(word_dictionary)
ratings_df = ratings_df.drop(['userId', 'timestamp'], axis=1)
ratings_df
ratings_df = ratings_df.groupby(ratings_df['movieId']).aggregate({'rating': 'mean'})

ratings_df
keywords_df
train_df = keywords_df.merge(ratings_df, on='movieId')
train_df
train_df = train_df.drop(train_df[train_df['keywords'] == '[]'].index, axis=0).reset_index(drop=True)
train_df['keywords'] = train_df['keywords'].apply(lambda word_list: [word_dict['id'] for word_dict in literal_eval(word_list)])
train_df
y = train_df.loc[:, 'rating']

X_raw = train_df.loc[:, 'keywords']
X_raw
y
word_counts = {}



for word_list in X_raw:

    for word in word_list:

        if word in word_counts:

            word_counts[word] += 1

        else:

            word_counts[word] = 1
word_counts
word_counts_sorted = {key: value for key, value in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}

word_counts_sorted
vocabulary = list(word_counts_sorted)[:100]

len(vocabulary)
X_raw
for word_list in X_raw:

    word_list[:] = [word for word in word_list if word in vocabulary]
X_raw
null_indices = set()



for i, words in enumerate(X_raw):

    if not words:

        null_indices.add(i)
X_raw = X_raw.drop(null_indices, axis=0).reset_index(drop=True)

y = y.drop(null_indices, axis=0).reset_index(drop=True)
X_raw
word_column_names = []



for word_list in X_raw:

    for word in word_list:

        if word not in word_column_names:

            word_column_names.append(word)
word_column_names = list(map(lambda x: word_dictionary[x], word_column_names))
mlb = MultiLabelBinarizer()



X = pd.DataFrame(mlb.fit_transform(X_raw), columns=word_column_names)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=24)
X.shape
inputs = tf.keras.Input(shape=(100,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='linear')(x)



model = tf.keras.Model(inputs, outputs)





model.compile(

    optimizer='adam',

    loss='mse'

)





batch_size = 32

epochs = 10



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()]

)
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'x': "Epoch", 'y': "Loss"},

    title="Loss Over Time"

)



fig.show()
np.argmin(history.history['val_loss'])
model.evaluate(X_test, y_test)
y_preds = model.predict(X_test)
y_test
y_preds
print(y_test.shape)

print(y_preds.shape)
y_test = y_test.to_numpy()

y_preds = np.squeeze(y_preds)
y_test
y_preds
rsquare = tfa.metrics.RSquare()



rsquare.update_state(y_test, y_preds)
print("R^2 Score:", rsquare.result().numpy())
pca = PCA(n_components=2)
X
X_reduced = pd.DataFrame(pca.fit_transform(X), columns=["PC1", "PC2"])

X_reduced
X
word_lists = []



for row in X_reduced.iterrows():

    word_list = [word for word in X.columns if X.loc[row[0], word] == 1]

    word_lists.append(word_list)
X_reduced['keywords'] = word_lists

X_reduced['keywords'] = X_reduced['keywords'].astype(str)
X_reduced
fig = px.scatter(

    X_reduced,

    x='PC1',

    y='PC2',

    hover_data={

        'PC1': False,

        'PC2': False,

        'keywords': True

    }

)



fig.show()