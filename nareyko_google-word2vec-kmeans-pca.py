import numpy as np
import pandas as pd
import re
import gc

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
%matplotlib inline

from tqdm import tqdm
tqdm.pandas()
news_df = pd.read_pickle('../input/data-preparation-memory-optimization/news_df.pkl')
news_df['date'] = news_df.time.dt.date
time_news = news_df.groupby('date').headline.apply(' '.join).reset_index()
del news_df; gc.collect()
# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
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
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)
time_news['words'] = time_news.headline.progress_apply(text_to_wordlist)
time_news.head(1).T
# Load Google pretrained model
model = KeyedVectors.load_word2vec_format('../input/word2vecnegative300/GoogleNews-vectors-negative300.bin', binary=True)
def text2vec(text):
    return np.mean([model[x] for x in text.split() if x in model.vocab], axis=0).reshape(1,-1)

time_news['vectors'] = time_news.words.progress_apply(text2vec)
time_news.head().T
time_news.to_pickle('time_news.pkl')
X = np.concatenate(time_news['vectors'].values)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
time_news['cluster'] = kmeans.predict(X)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
time_news['x'] = pca_result[:, 0]
time_news['y'] = pca_result[:, 1]
time_news.head()
cluster_colors = pd.np.array(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'])
time_news['color'] = cluster_colors[time_news.cluster.values]
time_news['text'] = time_news.headline.str[:50]

import bokeh.io
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet

# from bokeh.charts import Donut, HeatMap, Histogram, Line, Scatter, show, output_notebook, output_file
bokeh.io.output_notebook()
#visualize the data using bokeh
#output_file("top_artists.html", title="top artists")
# TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,previewsave"

source = ColumnDataSource.from_df(time_news[['x', 'y', 'color', 'text', 'date']])
TOOLTIPS = [("date", "@date"),("text", "@text")]
TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,previewsave"

plot = figure(plot_width=800, plot_height=450, tooltips=TOOLTIPS, tools=TOOLS)

#draw circles
plot.circle(y='y', x='x', source=source, size=15, fill_color='color')
show(plot)
