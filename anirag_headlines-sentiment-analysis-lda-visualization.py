# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/abcnews-date-text.csv",error_bad_lines=False,warn_bad_lines=False)

df.head()
df.shape
df.publish_date = pd.to_datetime(df.publish_date,format="%Y%m%d")
df.publish_date.min(),df.publish_date.max()
df.publish_date.max() - df.publish_date.min()
len(df.publish_date.unique())
s = df.groupby('publish_date').tail(2)
s.head()
all_headlines = s.headline_text.values
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

from nltk.corpus import stopwords

StopWords = stopwords.words("english")

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
%%time

sia = SIA()

pos_list = []

neg_list = []

neu_list = []

for post in all_headlines:

    post = " ".join([stemmer.stem(word) for word in str(post).lower().split() if word not in set(StopWords)])

    res = sia.polarity_scores(post)

    if res['compound'] > 0.0:

        pos_list.append(post)

    elif res['compound'] < 0.0:

        neg_list.append(post)

    else:

        neu_list.append(post)
print("Number of Positive Headlines : {}\nNumber of Negative Headlines : {}\nNumber of Neutral Headlines : {}".format(len(pos_list),len(neg_list),len(neu_list)))
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
pos_words = []

for line in pos_list:

    words = tokenizer.tokenize(line)

    for w in words:

        pos_words.append(w.lower())

    

    
neg_words = []

for line in neg_list:

    words = tokenizer.tokenize(line)

    for w in words:

        neg_words.append(w.lower())
from nltk import FreqDist

pos_words = FreqDist(pos_words)

for x in pos_words.most_common(10):

    print(x[0],":",x[1])
neg_words = FreqDist(neg_words)

for x in neg_words.most_common(10):

    print(x[0],":",x[1])
import matplotlib

import matplotlib.pylab as plt

%matplotlib inline

matplotlib.rcParams['xtick.labelsize'] = 14

plt.figure(figsize=(20,10))

pos_words.plot(50,cumulative=False)
plt.figure(figsize=(20,10))

neg_words.plot(50,cumulative=False)
sample = pos_list+neg_list+neu_list
import gensim

from gensim import corpora



sample_clean = [text.split() for text in sample] 
# Creating the term dictionary of our courpus, where every unique term is assigned an index. 

dictionary = corpora.Dictionary(sample_clean)



# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.

doc_term_matrix = [dictionary.doc2bow(doc) for doc in sample_clean]
%%time

# Creating the object for LDA model using gensim library

Lda = gensim.models.ldamodel.LdaModel

num_topics = 10

# Running and Trainign LDA model on the document term matrix.

ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=50,iterations=100)
dtm = ldamodel.get_document_topics(doc_term_matrix)

K = ldamodel.num_topics

topic_word_matrix = ldamodel.print_topics(K)
print("The topics are: \n")

for x in topic_word_matrix:

    print(x[0],":",x[1],"\n")
from gensim import matutils
document_topic_matrix = matutils.corpus2dense(corpus=dtm,num_docs=len(all_headlines),num_terms=K)
a = document_topic_matrix.transpose()
%%time

from sklearn.manifold import TSNE



# a t-SNE model

# angle value close to 1 means sacrificing accuracy for speed

# pca initializtion usually leads to better results 

tsne_model = TSNE(n_components=2, verbose=1, random_state=0,init='pca',)



# 8-D -> 2-D

tsne_lda = tsne_model.fit_transform(a)
_lda_keys = []

for i in range(a.shape[0]):

    _lda_keys.append(a[i].argmax())

len(_lda_keys)
import bokeh.plotting as bp

from bokeh.io import output_notebook

from bokeh.plotting import show



# 10 colors

colormap = np.array(["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c","#98df8a", "#d62728", "#ff9896","#bcbd22", "#dbdb8d"])

output_notebook()
plot_lda = bp.figure(plot_width=1000, plot_height=1000,

                     title="LDA t-SNE Viz",

                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

                     x_axis_type=None, y_axis_type=None, min_border=1)

n = len(a)

plot_lda.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1],

                 color=colormap[_lda_keys][:n],

                 source=bp.ColumnDataSource({

                   "content": sample_clean[:n],

                   "topic_key": _lda_keys[:n]

                   }))
topic_summaries = [x[1] for x in topic_word_matrix]

topic_coord = np.empty((a.shape[1], 2)) * np.nan

for topic_num in _lda_keys:

    topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]
# add topic words to graph

for i in range(a.shape[1]):

    plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])
show(plot_lda)