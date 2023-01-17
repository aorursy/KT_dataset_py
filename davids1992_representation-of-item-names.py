import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import FeatureUnion
import string

import IPython.display as ipd
import librosa.display

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
items_english = pd.read_csv("../input/predict-future-sales-supplementary/item_category.csv")

items = list(items["item_name"])
items_english = list(items_english["item_name_translated"])
_ = [print(items_english[i]) for i in range(10)] 
items_english = [''.join(sign for sign in item if sign not in string.punctuation) for item in items_english]
_ = [print(items_english[i]) for i in range(10)] 
np.random.seed(123)
ind = np.arange(len(items_english))
np.random.shuffle(ind)
ind = ind[:1000]
items_english = np.array(items_english)[ind]
tfidf_word_vectorizer = TfidfVectorizer(max_features=None, analyzer='word', ngram_range=(1, 1))
tfidf_word_features = tfidf_word_vectorizer.fit_transform(items_english)
len(tfidf_word_vectorizer.get_feature_names())
tfidf_char_vectorizer = TfidfVectorizer(max_features=None, analyzer='char', ngram_range=(3, 3))
tfidf_char_features = tfidf_char_vectorizer.fit_transform(items_english)
len(tfidf_char_vectorizer.get_feature_names())
projector = TSNE(n_components=2, perplexity=5, random_state=42)
tfidf_projection = projector.fit_transform(tfidf_char_features.toarray())
# Create a trace
trace = go.Scatter(
    text=items_english,
    x=tfidf_projection[:, 0],
    y=tfidf_projection[:, 1],
    mode='markers',
)
layout = go.Layout(title="Char tf-idf projection of items names", hovermode='closest')
figure = go.Figure(data=[trace], layout=layout)
py.iplot(figure, filename='projection.html')

fasttext_features = pd.read_csv("../input/predictfuturesalesitemnametranslatedfasttext/fasttext_features.csv")
fasttext_features.drop('item_id', axis=1, inplace=True)
fasttext_features.drop('Unnamed: 0', axis=1, inplace=True)
fasttext_features = fasttext_features.values
projector = TSNE(n_components=2, perplexity=60.0, random_state=42)
fasttext_projection = projector.fit_transform(fasttext_features)
fasttext_projection = fasttext_projection[ind]
# Create a trace
trace = go.Scatter(
    text=items_english,
    x=fasttext_projection[:, 0],
    y=fasttext_projection[:, 1],
    mode='markers',
)
layout = go.Layout(title="FastText projection of items names", hovermode='closest')
figure = go.Figure(data=[trace], layout=layout)
py.iplot(figure, filename='projection.html')
