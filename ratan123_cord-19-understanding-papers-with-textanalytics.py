!pip install textstat

!pip install chart_studio
import string

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

from collections import defaultdict

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

from tqdm import tqdm

%matplotlib inline





import nltk

nltk.download('punkt') # one time execution

import re

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize   

stop_words = stopwords.words('english')

remove_words = set(stopwords.words('english')) 



from plotly import tools

import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



import time

import pyLDAvis.sklearn

from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig

import textstat



import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from gensim.summarization.summarizer import summarize

from spacy.lang.en import English

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from statistics import *

import concurrent.futures



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



punctuations = string.punctuation

stopwords = list(STOP_WORDS)



parser = English()



def plot_readability(a,title,bins=0.1,colors=['#3A4750']):

    #reference and credits : https://www.kaggle.com/thebrownviking20/analyzing-quora-for-the-insinceres

    trace1 = ff.create_distplot([a], [" Abstract "], bin_size=bins, colors=colors, show_rug=False)

    trace1['layout'].update(title=title)

    iplot(trace1, filename='Distplot')

    table_data= [["Statistical Measures","Abstract"],

                ["Mean",mean(a)],

                ["Standard Deviation",pstdev(a)],

                ["Variance",pvariance(a)],

                ["Median",median(a)],

                ["Maximum value",max(a)],

                ["Minimum value",min(a)]]

    trace2 = ff.create_table(table_data)

    iplot(trace2, filename='Table')

    

punctuations = string.punctuation

stopwords = list(STOP_WORDS)



parser = English()

def spacy_tokenizer(sentence):

    #reference and credits : https://www.kaggle.com/thebrownviking20/analyzing-quora-for-the-insinceres

    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]

    mytokens = " ".join([i for i in mytokens])

    return mytokens





#references and credits : https://www.geeksforgeeks.org/print-colors-python-terminal/

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 

def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 

def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) 

def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) 

def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) 

def prCyan(skk): print("\033[96m {}\033[00m" .format(skk)) 

def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 

def prBlack(skk): print("\033[98m {}\033[00m" .format(skk)) 



import os

import warnings

warnings.filterwarnings('ignore')
#data set credits : https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv

biorxiv_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv')

clean_comm_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv')

clean_noncomm_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv')

clean_pmc_data = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv')
biorxiv_data.head(3)
clean_comm_data.head(3)
clean_noncomm_data.head(3)
clean_pmc_data.head(3)
from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=50, figure_size=(15.0,15.0), 

                   title = None, title_size=20, image_color=False,color = color):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color=color,

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(biorxiv_data['authors'].values, title="Word Cloud of Authors in biorxiv medrxiv Data",color = 'black')
plot_wordcloud(clean_comm_data['authors'].values, title="Word Cloud of Authors in comm use subset Data",color = 'white')
plot_wordcloud(clean_noncomm_data['authors'].values, title="Word Cloud of Authors in Non common use subset Data",color = 'red')
plot_wordcloud(clean_pmc_data['authors'].values, title="Word Cloud of Authors pmc_custom Data",color = 'violet')
plot_wordcloud(biorxiv_data['affiliations'].values, title="Word Cloud of Affiliations in biorxiv medrxiv Data ",color = 'green')
plot_wordcloud(clean_comm_data['affiliations'].values, title="Word Cloud of Affiliations in comm use subset Data",color = 'orange')
plot_wordcloud(clean_noncomm_data['affiliations'].values, title="Word Cloud of affiliations in Non common use subset Data",color = 'brown')
plot_wordcloud(clean_pmc_data['affiliations'].values, title="Word Cloud of Affiliations in pmc_custom Data",color = 'gray')
df1 = biorxiv_data['abstract'].dropna()

df3 = clean_comm_data["abstract"].dropna()

df2 = clean_noncomm_data["abstract"].dropna()

df4 = clean_pmc_data["abstract"].dropna()



## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    #Reference and credits: https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in df1:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in df2:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'orange')



freq_dict = defaultdict(int)

for sent in df3:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'black')



freq_dict = defaultdict(int)

for sent in df4:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(25), 'red')







# Creating two subplots

fig = tools.make_subplots(rows=2, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words in biorxiv_data", 

                                          "Frequent words in clean_comm_data",

                                          "Frequent words in clean_noncomm_data",

                                          "Frequent words in clean_pmc_data"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)







fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots of Abstracts")

iplot(fig, filename='word-plots')

freq_dict = defaultdict(int)

for sent in df1:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'gray')





freq_dict = defaultdict(int)

for sent in df2:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'orange')



freq_dict = defaultdict(int)

for sent in df3:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'brown')





freq_dict = defaultdict(int)

for sent in df4:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(25), 'pink')





# Creating two subplots

fig = tools.make_subplots(rows=2, cols=2, vertical_spacing=0.04,horizontal_spacing=0.25,

                          subplot_titles=["Frequent words in biorxiv_data", 

                                          "Frequent words in clean_comm_data",

                                          "Frequent words in clean_noncomm_data",

                                          "Frequent words in clean_pmc_data"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)





fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots of Abstracts")

iplot(fig, filename='word-plots')
for sent in df1:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'blue')





freq_dict = defaultdict(int)

for sent in df2:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'green')



freq_dict = defaultdict(int)

for sent in df3:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'violet')



freq_dict = defaultdict(int)

for sent in df4:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(25), 'red')









# Creating two subplots

fig = tools.make_subplots(rows=4, cols=1, vertical_spacing=0.04, horizontal_spacing=0.05,

                          subplot_titles=["Frequent words in biorxiv_data", 

                                          "Frequent words in clean_comm_data",

                                          "Frequent words in clean_noncomm_data",

                                          "Frequent words in clean_pmc_data"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)

fig.append_trace(trace3, 4, 1)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")

iplot(fig, filename='word-plots')


df1 = biorxiv_data['text'].dropna()

df3 = clean_comm_data["text"].dropna()

df2 = clean_noncomm_data["text"].dropna()

df4 = clean_pmc_data["text"].dropna()



## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in df1:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'orange')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in df2:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'green')



freq_dict = defaultdict(int)

for sent in df3:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'blue')



freq_dict = defaultdict(int)

for sent in df4:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(25), 'red')







# Creating two subplots

fig = tools.make_subplots(rows=2, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words in biorxiv_data", 

                                          "Frequent words in clean_comm_data",

                                          "Frequent words in clean_noncomm_data",

                                          "Frequent words in clean_pmc_data"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)







fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots of Texts in papers")

iplot(fig, filename='word-plots')

freq_dict = defaultdict(int)

for sent in df1:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'yellow')





freq_dict = defaultdict(int)

for sent in df2:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'red')



freq_dict = defaultdict(int)

for sent in df3:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'brown')





freq_dict = defaultdict(int)

for sent in df4:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(25), 'blue')





# Creating two subplots

fig = tools.make_subplots(rows=2, cols=2, vertical_spacing=0.04,horizontal_spacing=0.25,

                          subplot_titles=["Frequent words in biorxiv_data", 

                                          "Frequent words in clean_comm_data",

                                          "Frequent words in clean_noncomm_data",

                                          "Frequent words in clean_pmc_data"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)





fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots of Texts in papers")

iplot(fig, filename='word-plots')
%%time

text_1 = biorxiv_data["abstract"].dropna().apply(spacy_tokenizer)

text_2 = clean_comm_data["abstract"].dropna().apply(spacy_tokenizer)

text_3 = clean_noncomm_data['abstract'].dropna().apply(spacy_tokenizer)

text_4 = clean_pmc_data['abstract'].dropna().apply(spacy_tokenizer)

#count vectorization

vectorizer_1= CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

vectorizer_2= CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

vectorizer_3 =  CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

vectorizer_4= CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')



text1_vectorized = vectorizer_1.fit_transform(text_1)

text2_vectorized = vectorizer_2.fit_transform(text_2)

text3_vectorized = vectorizer_3.fit_transform(text_3)

text4_vectorized = vectorizer_4.fit_transform(text_4)
%%time

lda1 = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online',verbose=True)

lda2= LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online',verbose=True)

lda3 = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online',verbose=True)

lda4 = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online',verbose=True)



lda_1 = lda1.fit_transform(text1_vectorized)

lda_2 = lda2.fit_transform(text2_vectorized)

lda_3 = lda3.fit_transform(text3_vectorized)

lda_4 = lda4.fit_transform(text4_vectorized)
def selected_topics(model, vectorizer, top_n=10):

    for idx, topic in enumerate(model.components_):

        print("Topic %d:" % (idx))

        print([(vectorizer.get_feature_names()[i], topic[i])

                        for i in topic.argsort()[:-top_n - 1:-1]]) 
print("LDA Model of Bioarvix data Abstracts:")

selected_topics(lda1, vectorizer_1)
print("LDA Model of clean_comm data Abstracts:")

selected_topics(lda2, vectorizer_2)
print("LDA Model of clean_noncomm data Abstracts:")

selected_topics(lda3, vectorizer_3)
print("LDA Model of clean_pmc data Abstracts:")

selected_topics(lda4, vectorizer_4)
%%time

pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda1, text1_vectorized, vectorizer_1, mds='tsne')

dash
%%time

pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda2, text2_vectorized, vectorizer_2, mds='tsne')

dash
%%time

pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda3, text3_vectorized, vectorizer_3, mds='tsne')

dash
%%time

pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda4, text4_vectorized, vectorizer_4, mds='tsne')

dash
del text_1,text_2,text_3,text_4,vectorizer_1,vectorizer_2,vectorizer_3,vectorizer_4,text1_vectorized,text2_vectorized ,text3_vectorized ,text4_vectorized 
tqdm.pandas()

fre_1 = np.array(biorxiv_data["abstract"].dropna().apply(textstat.flesch_reading_ease))

plot_readability(fre_1,"Flesch Reading Ease",20)
tqdm.pandas()

fre_2 = np.array(clean_comm_data["abstract"].dropna().apply(textstat.flesch_reading_ease))

plot_readability(fre_2,"Flesch Reading Ease",20,colors = ['#8D99AE'] )
tqdm.pandas()

fre_3 = np.array(clean_noncomm_data["abstract"].dropna().apply(textstat.flesch_reading_ease))

plot_readability(fre_3,"Flesch Reading Ease",20,colors = ['#C65D17'])
tqdm.pandas()

fre_4 = np.array(clean_pmc_data["abstract"].dropna().apply(textstat.flesch_reading_ease))

plot_readability(fre_4,"Flesch Reading Ease",20,colors = ['#DDB967'])
dcr_ = np.array(biorxiv_data["abstract"].dropna().apply(textstat.dale_chall_readability_score))

plot_readability(dcr_,"Dale-Chall Readability Score",1,['#C65D17'])
dcr_ = np.array(clean_comm_data["abstract"].dropna().apply(textstat.dale_chall_readability_score))

plot_readability(dcr_,"Dale-Chall Readability Score",1,['#DDB967'])
dcr_ = np.array(clean_noncomm_data["abstract"].dropna().apply(textstat.dale_chall_readability_score))

plot_readability(dcr_,"Dale-Chall Readability Score",1,['#8D99AE'])
dcr_ = np.array(clean_pmc_data["abstract"].dropna().apply(textstat.dale_chall_readability_score))

plot_readability(dcr_,"Dale-Chall Readability Score",1,['#EF233C'])
ari_ = np.array(biorxiv_data["abstract"].dropna().apply(textstat.coleman_liau_index))

plot_readability(ari_,"Automated Readability Index",10,['#2B2D42'])
ari_ = np.array(clean_comm_data["abstract"].dropna().apply(textstat.coleman_liau_index))

plot_readability(ari_,"Automated Readability Index",10,['#FF934F'])
ari_ = np.array(clean_noncomm_data["abstract"].dropna().apply(textstat.coleman_liau_index))

plot_readability(ari_,"Automated Readability Index",10,['#488286'])
ari_ = np.array(clean_pmc_data["abstract"].dropna().apply(textstat.coleman_liau_index))

plot_readability(ari_,"Automated Readability Index",10,['#8491A3'])
cli_ = np.array(biorxiv_data["abstract"].dropna().apply(textstat.coleman_liau_index))

plot_readability(cli_,"The Coleman-Liau Index",10,['#e8434e'])
cli_ = np.array(clean_comm_data["abstract"].dropna().apply(textstat.coleman_liau_index))

plot_readability(cli_,"The Coleman-Liau Index",10,['#a36f72'])
cli_ = np.array(clean_noncomm_data["abstract"].dropna().apply(textstat.coleman_liau_index))

plot_readability(cli_,"The Coleman-Liau Index",10,['#2d5c5a'])
cli_ = np.array(clean_pmc_data["abstract"].dropna().apply(textstat.coleman_liau_index))

plot_readability(cli_,"The Coleman-Liau Index",10,['#64c48c'])
fog_ = np.array(biorxiv_data["abstract"].dropna().apply(textstat.gunning_fog))

plot_readability(fog_,"The Fog Scale (Gunning FOG Formula)",4,['#d98714'])
fog_ = np.array(clean_comm_data["abstract"].dropna().apply(textstat.gunning_fog))

plot_readability(fog_,"The Fog Scale (Gunning FOG Formula)",4,['#7dd609'])
fog_ = np.array(clean_noncomm_data["abstract"].dropna().apply(textstat.gunning_fog))

plot_readability(fog_,"The Fog Scale (Gunning FOG Formula)",4,['#E2D58B'])
fog_ = np.array(clean_pmc_data["abstract"].dropna().apply(textstat.gunning_fog))

plot_readability(fog_,"The Fog Scale (Gunning FOG Formula)",4,['#612620'])
#lets make one dataframe.

all_data = pd.concat([biorxiv_data,clean_comm_data,clean_noncomm_data,clean_pmc_data]).reset_index()
text = all_data['text'][10]

print('Text before summarizations:')

prCyan(text)
print('Text after summarization:')

prPurple(summarize(text,word_count = 250))