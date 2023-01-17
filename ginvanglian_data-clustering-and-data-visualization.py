import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import nltk

from nltk.stem.porter import *

import numpy as np

import pandas as pd 

import sklearn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import jaccard_similarity_score

cv = CountVectorizer()

from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



data = pd.read_csv('../input//questions.csv', encoding='ISO-8859-1')
data = pd.read_csv('../input/questions.csv').dropna()

len(data)
def cleaning(s):

    s = str(s)

    s = s.lower()

    s = re.sub('\s\W',' ',s)

    s = re.sub('\W,\s',' ',s)

    s = re.sub(r'[^\w]', ' ', s)

    s = re.sub("\d+", "", s)

    s = re.sub('\s+',' ',s)

    s = re.sub('[!@#$_]', '', s)

    s = s.replace("co","")

    s = s.replace("https","")

    s = s.replace(",","")

    s = s.replace("[\w*"," ")

    return s

data['question1'] = [cleaning(s) for s in data['question1']]

data['question2'] = [cleaning(s) for s in data['question2']]

  
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

from nltk.tokenize import sent_tokenize

data['processed_q1'] = data.apply(lambda row: nltk.word_tokenize(row['question1']),axis=1)

data['processed_q1'] = data['processed_q1'].apply(lambda x : [item for item in x if item not in stop])



data['processed_q2'] = data.apply(lambda row: nltk.word_tokenize(row['question2']),axis=1)

data['processed_q2'] = data['processed_q2'].apply(lambda x : [item for item in x if item not in stop])
data.shape
Duplicate_Questions = data.loc[(data.is_duplicate == 1)]

len(Duplicate_Questions)
Q1_D = pd.DataFrame(Duplicate_Questions)

Q1_Duplicate = Q1_D.question1

len(Q1_Duplicate)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

#vect = CountVectorizer(analyzer='word')

vectorizer = TfidfVectorizer(min_df=3, max_features = 10000)
tfid = vectorizer.fit_transform(Q1_Duplicate)

tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

len(tfidf)
from sklearn.decomposition import TruncatedSVD



svd = TruncatedSVD(n_components=50, random_state=0)

svd_tfidf = svd.fit_transform(tfid[:10000])
svd_tfidf.shape
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, random_state=0)

tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
import bokeh.plotting as bp

from bokeh.models import HoverTool, BoxSelectTool

from bokeh.plotting import figure, show, output_notebook



output_notebook()

plot_tfidf = bp.figure(plot_width=900, plot_height=700, title="Quora Dataset (tf-idf)",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)



plot_tfidf.scatter(x=tsne_tfidf[:,0], y=tsne_tfidf[:,1],

                    source=bp.ColumnDataSource({

                        "non_Proce": Q1_Duplicate[:3000], 

                        "processed":  Q1_Duplicate[:3000]

                    }))



hover = plot_tfidf.select(dict(type=HoverTool))

hover.tooltips={"non_Proce": "@non_Proce (processed: \"@processed\")"}

show(plot_tfidf)
from sklearn.cluster import MiniBatchKMeans



num_clusters = 6

kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 

                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)

kmeans = kmeans_model.fit(tfid)

kmeans_clusters = kmeans.predict(tfid)

kmeans_distances = kmeans.transform(tfid)
tsne_kmeans = tsne_model.fit_transform(kmeans_distances[:1000])
sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

for i in range(num_clusters):

    print("Cluster %d:" % i, end='')

    for j in sorted_centroids[i, :10]:

        print(' %s' % terms[j], end='')

    print()
import numpy as np



colormap = np.array([

    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", 

    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", 

    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", 

    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"

])



plot_kmeans = bp.figure(plot_width=800, plot_height=700, title="Quora Datset (k-means)",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)



plot_kmeans.scatter(x=tsne_kmeans[:,0], y=tsne_kmeans[:,1], 

                    color=colormap[kmeans_clusters][:10000], 

                    source=bp.ColumnDataSource({

                        "non_proce":Q1_Duplicate[:10000], 

                        "processed":Q1_Duplicate[:10000],

                        "cluster": kmeans_clusters[:10000]

                    }))

hover = plot_kmeans.select(dict(type=HoverTool))

hover.tooltips={"non_Proce": "@non_Proce (processed: \"@processed\"- cluster: @cluster)"}

show(plot_kmeans)
import gensim

from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer()

from gensim.models.doc2vec import Doc2Vec ,TaggedDocument

from nltk import tokenize
docs = [TaggedDocument(tknzr.tokenize(cleaning(data["question1"])), [i]) for i, tweet in enumerate(data)]
doc2vec_model = Doc2Vec(docs, size=100, window=8, min_count=5, workers=4)
doc2vec_model.most_similar(positive=["when", "how"], negative=["i"])