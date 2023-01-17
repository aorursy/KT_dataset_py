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
from os import path

import string



import matplotlib.pyplot as plt



from plotly import tools

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

print(__version__)

init_notebook_mode(connected=True)



import nltk

from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords



# Spacy

import spacy



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD



from sklearn.metrics.pairwise import cosine_distances

from sklearn.manifold import MDS

from sklearn.manifold import TSNE



from PIL import Image

from wordcloud import WordCloud, STOPWORDS

import urllib

from io import BytesIO
# NLTK 

wnl = WordNetLemmatizer()

sb_stemmer = SnowballStemmer("english")



# Spacy

nlp = spacy.load('en')



# Plotly

author_images = [

    "https://upload.wikimedia.org/wikipedia/commons/8/84/Edgar_Allan_Poe_daguerreotype_crop.png",

    "https://upload.wikimedia.org/wikipedia/commons/6/65/RothwellMaryShelley.jpg",

    "https://upload.wikimedia.org/wikipedia/commons/1/10/H._P._Lovecraft%2C_June_1934.jpg"

]



paper_bgcolor="rgb(240,240,240)"

plot_bgcolor="rgb(240,240,240)"



# color definitions (from colorlover module)

# import colorlover as cl

# cl.scales["3"]["div"]["RdBu"]

rd_bu = ['rgb(239,138,98)', 'rgb(247,247,247)', 'rgb(103,169,207)']
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
def shrink_train_data_size_by_factor(factor, text_arr):

    text_arr_reduced = []

    for i in range(len(text_arr) // factor):

        start = i * factor

        end = (start + factor) - 1

        text_arr_reduced.append("\n".join(text_arr[start:end+1]))

    if len(text_arr) % factor != 0:

        rem_elments = len(text_arr) % factor

        start = (len(text_arr) // factor) * factor

        end = (start + rem_elments) - 1

        text_arr_reduced.append("\n".join(text_arr[start:end+1]))

    return text_arr_reduced
counts_by_author = train["author"].value_counts()

author_names = list(counts_by_author.index)

instances_per_author = list(counts_by_author)



# make copy of the original training data set

counts_by_author_orig = list(counts_by_author)

author_names_orig = list(author_names)

instances_per_author_orig = list(instances_per_author)

train_orig = train.copy()



# factor to which the train data should be reduced

factor = 20



# temporary variables to hold the reduced data

shrinked_train = []

shrinked_labels = []



for i in range(len(author_names)):

    instances = train[train["author"] == author_names[i]]["text"].as_matrix()

    instances_red = shrink_train_data_size_by_factor(factor, instances)

    labels_red = [author_names[i]] * len(instances_red)

    shrinked_train += instances_red

    shrinked_labels += labels_red



# training data (shrinked)

train = pd.DataFrame({"text": shrinked_train, "author": shrinked_labels})
counts_by_author = train["author"].value_counts()

author_names = list(counts_by_author.index)

instances_per_author = list(counts_by_author)

print("Original training data stats")

print("----------------------------")

print("Training data size: ", train_orig.shape)

print("Author names:", author_names_orig)

print("Instances per author:", instances_per_author_orig)



print("\nAfter shrinking the training data by factor: {}".format(factor))

print("------------------------------------------------")

print("Training data size: ", train.shape)

print("Author names:", author_names)

print("Instances per author:", instances_per_author)

train.head()
# concatenate texts belonging to same author

combined_texts = []

for author in author_names: 

    texts_of_author = train[train["author"] == author]["text"]

    texts_np_array = texts_of_author.as_matrix()

    text_together = " ".join(texts_np_array)

    combined_texts.append(text_together)

combined_texts = np.asarray(combined_texts)
layout_images = []

for i in range(len(author_images)):

    layout_image = dict( source = author_images[i], 

                        xref="paper", 

                        yref="paper", 

                        x= (instances_per_author_orig[i] / max(instances_per_author_orig)) - 0.15, 

                        y = (i / len(author_images))+ 0.05, 

                        sizex=0.4,

                        sizey=0.2,

                        xanchor="left", 

                        yanchor="bottom")

    layout_images.append(layout_image)

        

bar_data = [go.Bar(

            x=instances_per_author_orig,

            y=author_names_orig,

            orientation = 'h'

)]

layout = go.Layout(title = "Training data distribution (original data)", images=layout_images,

    xaxis=dict(title="# of training instances"),

    yaxis=dict(title="Author"))

fig = dict(data=bar_data, layout=layout)

iplot(fig)
layout_images = []

for i in range(len(author_images)):

    layout_image = dict( source = author_images[i], 

                        xref="paper", 

                        yref="paper", 

                        x= (instances_per_author[i] / max(instances_per_author)) - 0.15, 

                        y = (i / len(author_images))+ 0.05, 

                        sizex=0.4,

                        sizey=0.2,

                        xanchor="left", 

                        yanchor="bottom")

    layout_images.append(layout_image)

        

bar_data = [go.Bar(

            x=instances_per_author,

            y=author_names,

            orientation = 'h'

)]

layout = go.Layout(title = "Training data distribution (after shrinking)", images=layout_images,

    xaxis=dict(title="# of training instances"),

    yaxis=dict(title="Author"))

fig = dict(data=bar_data, layout=layout)

iplot(fig)
stop_words_en = set(stopwords.words('english'))
def avg_sentence_length_for_text(text):

    """Given a text containing one more sentences in the form of 

    paragraph, the function returns the average sentence length

    for the entire paragraph.

    """

    sentences = nltk.sent_tokenize(text)

    num_tokens_in_corpus = 0

    for s in sentences:

        num_tokens_in_corpus += len(nltk.word_tokenize(s))

    return float(num_tokens_in_corpus) / len(sentences)
def avg_stop_words_per_sentence(text):

    """Given a text containing one more sentences in the form of 

    paragraph, the function returns the average number of function words

    per sentence in the given text

    """    

    sentences = nltk.sent_tokenize(text)

    num_stop_words_in_corpus = 0

    for s in sentences:

        s_tokens =  nltk.word_tokenize(s.lower())

        for w in s_tokens:

            if w in stop_words_en: 

                num_stop_words_in_corpus += 1

    return float(num_stop_words_in_corpus) / len(sentences)
def make_array_size_divisible_by_factor(oned_nparray, fc):

    """Given an 1d array of an arbitrary size, make the shape of the 

    1d array divisible by 100 by appending np.nan values. For ex: If the array 

    dimension is (114, ) then the np.nan 1d array of shape 86 will 

    be appended to the original 1d array to become array size 200. 

    """

    remainder = oned_nparray.shape[0] % fc

    if remainder > 0:

        cells_to_fill = fc - remainder

        nan_array = np.full(cells_to_fill, np.nan)

        oned_nparray = np.append(oned_nparray, nan_array)

    num_cols_heatmap = int(oned_nparray.shape[0] / fc)

    num_rows_heatmap = fc

    return oned_nparray.reshape((num_rows_heatmap, num_cols_heatmap))
def get_reshaped_sen_len_trace_for_author(author_name):

    sen_lengths_for_author = train_copy[train_copy["author"] == author_name]

    sen_lengths_nparray = sen_lengths_for_author["sen_len"].as_matrix()

    sen_lengths_nparray_reshaped = make_array_size_divisible_by_factor(sen_lengths_nparray, 25)    

    return sen_lengths_nparray_reshaped
def get_reshaped_stop_words_trace_for_author(author_name):

    stop_words_for_author = train_copy[train_copy["author"] == author_name]

    stop_words_counts_nparray = stop_words_for_author["stop_words"].as_matrix()

    stop_words_counts_nparray_reshaped = make_array_size_divisible_by_factor(stop_words_counts_nparray, 25)    

    return stop_words_counts_nparray_reshaped
train_copy = train.copy()

train_copy["sen_len"] = train_copy["text"].apply(avg_sentence_length_for_text)

train_copy["stop_words"] = train_copy["text"].apply(avg_stop_words_per_sentence)
avg_sen_len_used_by_authors = []

for author in author_names:

    temp = train_copy[train_copy["author"] == author]

    num_tokens = temp["sen_len"].sum()

    avg_sen_len = float(num_tokens) / temp.shape[0]

    avg_sen_len_used_by_authors.append(avg_sen_len)
fig_coords = [(1,1), (1,2), (1,3)]

axes_names = [("x1", "y1"), ("x2", "y2"), ("x3", "y3")]

axes_lo_names = [("xaxis1", "yaxis1"), ("xaxis2", "yaxis2"), ("xaxis3", "yaxis3")]

fig = tools.make_subplots(rows=1, cols=3, subplot_titles=(author_names[0], author_names[1], author_names[2]))

for author, fig_coord, ax in zip(author_names, fig_coords, axes_names):

    reshaped_sen_lengths = get_reshaped_sen_len_trace_for_author(author)

    trace = go.Heatmap(z=reshaped_sen_lengths, colorscale = 'Portland', zmin=0, zmax=80, xaxis=ax[0], yaxis=ax[1])

    fig.append_trace(trace, fig_coord[0], fig_coord[1])

fig["layout"].update(title = "Average sentence lengths (# of words)")

for ax_name in axes_lo_names:

    fig["layout"][ax_name[0]].update(showgrid=False, showline=False, zeroline=False, ticks='', showticklabels=False)

    fig["layout"][ax_name[1]].update(showgrid=False, showline=False, zeroline=False, ticks='', showticklabels=False)    

iplot(fig)



fig_coords = [(1,1), (1,2), (1,3)]

axes_names = [("x1", "y1"), ("x2", "y2"), ("x3", "y3")]

axes_lo_names = [("xaxis1", "yaxis1"), ("xaxis2", "yaxis2"), ("xaxis3", "yaxis3")]

fig = tools.make_subplots(rows=1, cols=3, subplot_titles=(author_names[0], author_names[1], author_names[2]))

for author, fig_coord, ax in zip(author_names, fig_coords, axes_names):

    reshaped_stop_word_counts = get_reshaped_stop_words_trace_for_author(author)

    trace = go.Heatmap(z=reshaped_stop_word_counts, colorscale = 'Portland', zmin=0, zmax=35, xaxis=ax[0], yaxis=ax[1])

    fig.append_trace(trace, fig_coord[0], fig_coord[1])

fig["layout"].update(title = "Average number of stop words per sentence")

for ax_name in axes_lo_names:

    fig["layout"][ax_name[0]].update(showgrid=False, showline=False, zeroline=False, ticks='', showticklabels=False)

    fig["layout"][ax_name[1]].update(showgrid=False, showline=False, zeroline=False, ticks='', showticklabels=False)

iplot(fig)
counts_vectorizer = CountVectorizer(stop_words="english", min_df=3)

counts_comb = counts_vectorizer.fit_transform(combined_texts)



tfidf_comb = TfidfTransformer().fit_transform(counts_comb)



cosine_dist = cosine_distances(tfidf_comb)

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(cosine_dist)

xs, ys = pos[:,0], pos[:, 1]

data = []

for i in range(len(author_names)): 

    trace = go.Scatter(x=[xs[i]], y=[ys[i]], mode="markers",

                       marker= dict(size= 20, line= dict(width=1), color= rd_bu[i]),

                       name= author_names[i])

    data.append(trace)

layout = go.Layout(title="How close authors are to each other?")

fig = go.Figure(data=data, layout=layout)

iplot(fig)
def is_valid_token(tok):

    """The function returns false, if 

    1. tok length is < 3

    2. tok contains non-alphabetic characters

    """

    if not tok.isalpha(): 

        return False 

    if len(tok) < 4: 

        return False

    return True
def lemmatize(input_str):

    tokens = word_tokenize(input_str)

    tokens = [t for t in tokens if is_valid_token(t) is True]

    lemmatized = [wnl.lemmatize(t) for t in tokens]

    return lemmatized
num_topics = 20



vectorizer_ind = CountVectorizer(stop_words="english", min_df=5, tokenizer=lemmatize)

counts_ind = vectorizer_ind.fit_transform(train["text"])

vocab_ind = np.array(vectorizer_ind.get_feature_names())

tfidf_ind = TfidfTransformer().fit_transform(counts_ind)



clf = NMF(n_components=num_topics)

doctopic = clf.fit_transform(tfidf_ind)
# words associated with topics

topic_strings = []

for topic in clf.components_:

    word_idx = np.argsort(topic)[::-1][0:10]

    topic_words = [vocab_ind[i] for i in word_idx]

    topic_strings.append(" ".join(topic_words))

    #topic_words.append([vocab_ind[i] for i in word_idx])



authors_of_train_data = train["author"].values

doctopic_one_per_author = np.zeros((len(author_names), num_topics))

for i in range(len(author_names)): 

    doctopic_one_per_author[i, :] = np.mean(doctopic[authors_of_train_data == author_names[i], :], axis=0)



doctopic_one_per_author = doctopic_one_per_author / np.sum(doctopic_one_per_author, axis=1, keepdims=True)



for t in range(len(topic_strings)):

    print("Topic {}: {}".format(t+1, topic_strings[t]))
topic_headers = [ "topic-"+str(i) for i in range(1, num_topics+1)]

trace = go.Heatmap(z=doctopic_one_per_author.T, x = author_names, y=topic_strings, colorscale='Portland')

layout = go.Layout(height=600, 

                   width=900,

                   margin=go.Margin(l=400, t=50, r=150, b=200), title="Heatmap of topics")

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
tsne_doctopic_model = TSNE(n_components=2)

tsne_doctopic_output = tsne_doctopic_model.fit_transform(doctopic)



svd_doctopic_model = TruncatedSVD(n_components=2)

svd_doctopic_output = svd_doctopic_model.fit_transform(doctopic)
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=("SVD", "TSNE"))

fig_coords = [(1,1), (1,2)]

for i in range(len(author_names)):

    x_points = svd_doctopic_output[authors_of_train_data == author_names[i], 0]

    y_points = svd_doctopic_output[authors_of_train_data == author_names[i], 1]

    trace = go.Scatter(x=x_points, y=y_points, mode="markers",

                           marker= dict(size= 10, line= dict(width=0.5), color= rd_bu[i]),

                           name= author_names[i])

    fig.append_trace(trace, 1, 1)

for i in range(len(author_names)):

    x_points = tsne_doctopic_output[authors_of_train_data == author_names[i], 0]

    y_points = tsne_doctopic_output[authors_of_train_data == author_names[i], 1]

    trace = go.Scatter(x=x_points, y=y_points, mode="markers",

                           marker= dict(size= 10, line= dict(width=0.5), color= rd_bu[i]),

                           name= author_names[i], showlegend=False)

    fig.append_trace(trace, 1, 2)

fig['layout'].update(height=500, width=800, title="Visualizing training instances in reduced dimension")

iplot(fig)