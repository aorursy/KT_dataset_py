import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import warnings, time, gc
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)
color = sns.color_palette("Set2")
warnings.filterwarnings("ignore")
%matplotlib inline

from kaggle.competitions import twosigmanews

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

env = twosigmanews.make_env()
market_train, news_train = env.get_training_data()
del market_train; gc.collect()
news_train.head()
news_train["headline_len"] = news_train["headline"].apply(lambda x: len(x))
news_train["headline_len"].hist(figsize = (15, 5), bins = 100)
plt.show()
max_len = max(news_train["headline_len"])
temp = news_train[news_train["headline_len"] == max_len]["headline"]
for t in temp:
    print(t)
    print()
temp = news_train[news_train["headline_len"] == 0]
temp.head()
# Drop rows with empty headlines
news_train.drop(news_train[news_train["headline_len"] == 0].index, 
                inplace = True)
news_train.head()
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer 
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

import re
from functools import reduce

stop_words = set(stopwords.words("english"))
#from:https://github.com/RenatoBMLR/nlpPy/tree/master/src

class TextDataset():

    def __init__(self, df, lang = 'english'):

        self.data = df

        self.tokenizer = TweetTokenizer()
        self.stop_words = set(stopwords.words(lang))
        self.lemmatizer = WordNetLemmatizer()
        self.ps = PorterStemmer()
        
    def _get_tokens(self, words):    
        return [word.lower() for word in words.split()]
    
    def _removeStopwords(self, words):
        # Removing all the stopwords
        return [word for word in words if word not in self.stop_words]

    def _removePonctuation(self, words):
        return re.sub(r'[^\w\s]', '', words)

    def _lemmatizing(self, words):
        #Lemmatizing
        return [self.lemmatizer.lemmatize(word) for word in words]

    def _stemming(self, words):
        #Stemming
        return [self.ps.stem(word) for word in words]


    def process_data(self, col = 'content', remove_pontuation=True, remove_stopw = True, lemmalize = False, stem = False):

        self.data = self.data.drop_duplicates(subset=col, keep="last")
        
        proc_col = col
        if remove_pontuation:
            proc_col = col + '_data'
            self.data[proc_col] = self.data[col].apply(lambda x: self._removePonctuation(x) )
        
        # get tokens of the sentence
        self.data[proc_col] = self.data[proc_col].apply(lambda x: self._get_tokens(x))
        if remove_stopw:
            self.data[proc_col] = self.data[proc_col].apply(lambda x: self._removeStopwords(x)) 
        if lemmalize:
            self.data[proc_col] = self.data[proc_col].apply(lambda x: self._lemmatizing(x) )
        if stem:
            self.data[proc_col] = self.data[proc_col].apply(lambda x: self._stemming(x))

        self.data['nb_words'] = self.data[proc_col].apply(lambda x: len(x))
        self.proc_col = proc_col
        
    def __len__(self):
        return len(self.data)
partial_news = news_train[news_train["time"] >= "2016-12-01"]
text_tokens = TextDataset(partial_news)
text_tokens.process_data(col = "headline")
text_tokens.data["headline_data"].head()
partial_news["tokens"] = text_tokens.data["headline_data"]
partial_news.head()
partial_news.fillna(method = "bfill", inplace = True)
tf_idf_vec = TfidfVectorizer(min_df = 3, 
                             max_features = 100000, 
                             analyzer = "word",
                             ngram_range = (1, 2),
                             stop_words = "english")
tf_idf = tf_idf_vec.fit_transform(list(partial_news["tokens"].map(lambda tokens: " ".join(tokens))))
tfidf_df = dict(zip(tf_idf_vec.get_feature_names(), tf_idf_vec.idf_))
tfidf_df = pd.DataFrame(columns = ["tfidf"]).from_dict(dict(tfidf_df), orient = "index")
tfidf_df.columns = ["tfidf"]
tfidf_df.tfidf.hist(bins = 25, figsize = (15, 5))
plt.show()
tfidf_df.sort_values(by = ["tfidf"], ascending = True).head(10)
tfidf_df.sort_values(by=["tfidf"], ascending = False).head(10)
tf_idf.shape
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components = 30, random_state = 32)
svd_tfidf = svd.fit_transform(tf_idf)

svd_tfidf.shape
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components = 2, verbose = 1, random_state = 32, n_iter = 500)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
tsne_tfidf.shape
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook, reset_output
from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file
tsne_tfidf_df = pd.DataFrame(tsne_tfidf)
tsne_tfidf_df.columns = ["x", "y"]
tsne_tfidf_df["asset_name"] = partial_news["assetName"].values
tsne_tfidf_df["headline"] = partial_news["headline"].values
tsne_tfidf_df.head()
output_notebook()
plot_tfidf = bp.figure(plot_width = 700, plot_height = 600, 
                       title = "tf-idf clustering of stock market news",
                       tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                       x_axis_type = None, y_axis_type = None, min_border = 1)

# palette = d3["Category10"][len(tsne_tfidf_df["asset_name"].unique())]
# color_map = bmo.CategoricalColorMapper(factors = tsne_tfidf_df["asset_name"].map(str).unique(), 
#                                        palette = palette)

plot_tfidf.scatter(x = "x", y = "y", 
#                    color = {"field": "asset_name", "transform": color_map}, 
#                    legend = "asset_name",
                   source = tsne_tfidf_df,
                   alpha = 0.7)
hover = plot_tfidf.select(dict(type = HoverTool))
hover.tooltips = {"headline": "@headline", "asset_name": "@asset_name"}

show(plot_tfidf)
from sklearn.cluster import MiniBatchKMeans

kmeans_model = MiniBatchKMeans(n_clusters = 50, # don't have time to find the best number
                               init = "k-means++",
                               n_init =  1,
                               init_size = 1000, 
                               batch_size = 1000, 
                               verbose = 0, 
                               max_iter = 1000)
kmeans = kmeans_model.fit(tf_idf)
kmeans_clusters = kmeans.predict(tf_idf)
kmeans_distances = kmeans.transform(tf_idf)
tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
tsne_kmeans_df = pd.DataFrame(tsne_kmeans)
tsne_kmeans_df.columns = ["x", "y"]
tsne_kmeans_df["cluster"] = kmeans_clusters
tsne_kmeans_df["asset_name"] = partial_news["assetName"].values
tsne_kmeans_df["headline"] = partial_news["headline"].values
colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5", "#e3be38", 
                     "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
                     "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", 
                     "#d07d3c", "#52697d", "#194196", "#d27c88", "#36422b", "#b68f79", "#00ffff", "#33ff33",
                     "#ffff99", "#99ff33", "#ff6666", "#666600", "#99004c", "#808080", "#a80a0a", "#a4924c",
                     "#4a8e92", "#92734a", "#7d4097", "#4b4097", "#c0c0c0", "#409794", "#1a709b", "#a7dcf6",
                     "#b1a7f6", "#eea7f6"])
plot_kmeans = bp.figure(plot_width = 700, plot_height = 600, 
                       title = "k-means clustering of stock market news",
                       tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                       x_axis_type = None, y_axis_type = None, min_border = 1)

source = ColumnDataSource(data = dict(x = tsne_kmeans_df["x"], y = tsne_kmeans_df["y"],
                                      color = colormap[kmeans_clusters],
                                      headline = tsne_kmeans_df["headline"],
                                      asset_name = tsne_kmeans_df["asset_name"],
                                      cluster = tsne_kmeans_df["cluster"]))

plot_kmeans.scatter(x = "x", y = "y", color = "color", source = source)
hover = plot_kmeans.select(dict(type = HoverTool))
hover.tooltips = {"headline": "@headline", "asset_name": "@asset_name", "cluster": "@cluster"}
show(plot_kmeans)
cv = CountVectorizer(min_df = 2,
                     max_features = 100000,
                     analyzer = "word",
                     ngram_range = (1, 2),
                     stop_words = "english")
count_vectors = cv.fit_transform(partial_news["headline"])
lda_model = LatentDirichletAllocation(n_components = 20, 
                                      # we choose a small n_components for time convenient
                                      learning_method = "online",
                                      max_iter = 20,
                                      random_state = 32)
news_topics = lda_model.fit_transform(count_vectors)
n_top_words = 10
topic_summaries = []
topic_word = lda_model.components_
vocab = cv.get_feature_names()

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(" ".join(topic_words))
    print("Topic {}: {}".format(i, " | ".join(topic_words)))
tsne_lda = tsne_model.fit_transform(news_topics)
news_topics = np.matrix(news_topics)
doc_topics = news_topics/news_topics.sum(axis = 1)

lda_keys = []
for i, tweet in enumerate(partial_news["headline"]):
    lda_keys += [doc_topics[i].argmax()]
    
tsne_lda_df = pd.DataFrame(tsne_lda, columns = ["x", "y"])
tsne_lda_df["headline"] = partial_news["headline"].values
tsne_lda_df["asset_name"] = partial_news["assetName"].values
tsne_lda_df["topics"] = lda_keys
tsne_lda_df["topics"] = tsne_lda_df["topics"].map(int)
plot_lda = bp.figure(plot_width = 700, plot_height = 600, 
                    title = "LDA topics of stock market news",
                    tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                    x_axis_type = None, y_axis_type = None, min_border = 1)

source = ColumnDataSource(data = dict(x = tsne_lda_df["x"], y = tsne_lda_df["y"],
                         color = colormap[lda_keys],
                         headline = tsne_lda_df["headline"],
                         asset_name = tsne_lda_df["asset_name"],
                         topics = tsne_lda_df["topics"]))

plot_lda.scatter(x = "x", y = "y", color = "color", source = source)
hover = plot_lda.select(dict(type = HoverTool))
hover.tooltips = {"headline": "@headline", "asset_name": "@asset_name", "topics": "@topics"}
show(plot_lda)
