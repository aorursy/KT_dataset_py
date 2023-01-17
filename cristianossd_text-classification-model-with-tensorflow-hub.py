%env JOBLIB_TEMP_FOLDER=/tmp
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.preprocessing import MultiLabelBinarizer
DATASET_FILE='../input/movies_metadata.csv'
df = pd.read_csv(DATASET_FILE)
df.fillna('', inplace=True)
df.head(2)
import json


descriptions = df['overview'].values
genres = df['genres'].apply(lambda genre: list(map(lambda obj: obj['name'],json.loads(genre.replace('\'', '"'))))).values
train_size = int(len(descriptions) * 0.8)

train_descriptions = descriptions[:train_size]
train_genres = genres[:train_size]

test_descriptions = descriptions[train_size:]
test_genres = genres[train_size:]
descriptions_embeddings = hub.text_embedding_column(
    'descriptions',
    module_spec='https://tfhub.dev/google/universal-sentence-encoder/2')
top_genres = ['Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary', 'Adventure', 'Science Fiction']
[1, 0, 0, 0, 0, 0, 0, 1, 0]  # multi-hot label for a comedy and adventure movie
encoder = MultiLabelBinarizer()
encoder.fit_transform(train_genres)
train_encoded = encoder.transform(train_genres)
test_encoded = encoder.transform(test_genres)
num_classes = len(encoder.classes_)
num_classes
multi_label_head = tf.contrib.estimator.multi_label_head(
    num_classes,
    loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
estimator = tf.contrib.estimator.DNNEstimator(
    head=multi_label_head,
    hidden_units=[64, 10],
    feature_columns=[descriptions_embeddings])
labels = np.array(train_encoded)
features = {
    'descriptions': np.array(train_descriptions)
}

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    features,
    labels,
    shuffle=True,
    batch_size=32,
    num_epochs=20)
estimator.train(input_fn=train_input_fn)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'descriptions': np.array(test_descriptions).astype(np.str)},
    test_encoded.astype(np.int32),
    shuffle=False)
estimator.evaluate(input_fn=eval_input_fn)
# from IMBD
raw_test = [
    "An examination of our dietary choices and the food we put in our bodies. Based on Jonathan Safran Foer's memoir.", # Documentary
    "A teenager tries to survive the last week of her disastrous eighth-grade year before leaving to start high school.", # Comedy
    "Ethan Hunt and his IMF team, along with some familiar allies, race against time after a mission gone wrong." # Action, Adventure
]

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'descriptions': np.array(raw_test).astype(np.str)},
    shuffle=False)
prediction = estimator.predict(predict_input_fn)
for movie_genres in prediction:
    top_2 = movie_genres['probabilities'].argsort()[-2:][::-1]
    for genre in top_2:
        text_genre = encoder.classes_[genre]
        print(text_genre + ': ' + str(round(movie_genres['probabilities'][genre] * 100, 2)) + '%')
