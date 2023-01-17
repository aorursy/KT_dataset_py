import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import warnings



from keras.layers import Input, Embedding, Flatten, Dot, Dense

from keras.models import Model



warnings.filterwarnings('ignore')

%matplotlib inline
dataset = pd.read_csv('../input/ratings.csv')
dataset.head()
dataset.shape
from sklearn.model_selection import train_test_split

train, test = train_test_split(dataset, test_size=0.2, random_state=42)
train.head()
test.head()
n_users = len(dataset.user_id.unique())

n_users
n_books = len(dataset.book_id.unique())

n_books
# creating book embedding path

book_input = Input(shape=[1], name="Book-Input")

book_embedding = Embedding(n_books+1, 5, name="Book-Embedding")(book_input)

book_vec = Flatten(name="Flatten-Books")(book_embedding)



# creating user embedding path

user_input = Input(shape=[1], name="User-Input")

user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)

user_vec = Flatten(name="Flatten-Users")(user_embedding)



# performing dot product and creating model

prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec])

model = Model([user_input, book_input], prod)

model.compile('adam', 'mean_squared_error')
from keras.models import load_model



if os.path.exists('regression_model.h5'):

    model = load_model('regression_model.h5')

else:

    history = model.fit([train.user_id, train.book_id], train.rating, epochs=5, verbose=1)

    model.save('regression_model.h5')

    plt.plot(history.history['loss'])

    plt.xlabel("Epochs")

    plt.ylabel("Training Error")
model.evaluate([test.user_id, test.book_id], test.rating)
predictions = model.predict([test.user_id.head(10), test.book_id.head(10)])



[print(predictions[i], test.rating.iloc[i]) for i in range(0,10)]
from keras.layers import Concatenate



# creating book embedding path

book_input = Input(shape=[1], name="Book-Input")

book_embedding = Embedding(n_books+1, 5, name="Book-Embedding")(book_input)

book_vec = Flatten(name="Flatten-Books")(book_embedding)



# creating user embedding path

user_input = Input(shape=[1], name="User-Input")

user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)

user_vec = Flatten(name="Flatten-Users")(user_embedding)



# concatenate features

conc = Concatenate()([book_vec, user_vec])



# add fully-connected-layers

fc1 = Dense(128, activation='relu')(conc)

fc2 = Dense(32, activation='relu')(fc1)

out = Dense(1)(fc2)



# Create model and compile it

model2 = Model([user_input, book_input], out)

model2.compile('adam', 'mean_squared_error')
from keras.models import load_model



if os.path.exists('regression_model2.h5'):

    model2 = load_model('regression_model2.h5')

else:

    history = model2.fit([train.user_id, train.book_id], train.rating, epochs=5, verbose=1)

    model2.save('regression_model2.h5')

    plt.plot(history.history['loss'])

    plt.xlabel("Epochs")

    plt.ylabel("Training Error")
model2.evaluate([test.user_id, test.book_id], test.rating)
predictions = model2.predict([test.user_id.head(10), test.book_id.head(10)])



[print(predictions[i], test.rating.iloc[i]) for i in range(0,10)]
# Extract embeddings

book_em = model.get_layer('Book-Embedding')

book_em_weights = book_em.get_weights()[0]
book_em_weights[:5]
from sklearn.decomposition import PCA

import seaborn as sns



pca = PCA(n_components=2)

pca_result = pca.fit_transform(book_em_weights)

sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])
book_em_weights = book_em_weights / np.linalg.norm(book_em_weights, axis = 1).reshape((-1, 1))

book_em_weights[0][:10]

np.sum(np.square(book_em_weights[0]))
pca = PCA(n_components=2)

pca_result = pca.fit_transform(book_em_weights)

sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tnse_results = tsne.fit_transform(book_em_weights)
sns.scatterplot(x=tnse_results[:,0], y=tnse_results[:,1])
# Creating dataset for making recommendations for the first user

book_data = np.array(list(set(dataset.book_id)))

book_data[:5]
user = np.array([1 for i in range(len(book_data))])

user[:5]
predictions = model.predict([user, book_data])



predictions = np.array([a[0] for a in predictions])



recommended_book_ids = (-predictions).argsort()[:5]



recommended_book_ids
# print predicted scores

predictions[recommended_book_ids]
books = pd.read_csv('../input/books.csv')

books.head()
books[books['id'].isin(recommended_book_ids)]