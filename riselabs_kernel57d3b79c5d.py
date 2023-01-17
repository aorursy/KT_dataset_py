!pip install chart_studio
!pip install textstat
!pip install pyLDAvis
import os

import json

from pprint import pprint

from copy import deepcopy



import numpy as np  # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.notebook import tqdm

import string

import random

from collections import defaultdict

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



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

from spacy.lang.en import English

parser = English()
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

filenames = os.listdir(data_dir)

print("Number of articles retrieved from biorxiv:", len(filenames))
all_files = []



for filename in filenames:

    filename = data_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)
file = all_files[0]

print("Dictionary keys:", file.keys())
def format_name(author):

    middle_name = " ".join(author['middle'])

    

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])
def format_affiliation(affiliation):

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))

    

    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)
def format_authors(authors, with_affiliation=False):

    name_ls = []

    

    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)

    

    return ", ".join(name_ls)
def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body
def format_bib(bibs):

    if type(bibs) == dict:

        bibs = list(bibs.values())

    bibs = deepcopy(bibs)

    formatted = []

    

    for bib in bibs:

        bib['authors'] = format_authors(

            bib['authors'], 

            with_affiliation=False

        )

        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]

        formatted.append(", ".join(formatted_ls))



    return "; ".join(formatted)
def load_files(dirname):

    filenames = os.listdir(dirname)

    raw_files = []



    for filename in tqdm(filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)

    

    return raw_files
def generate_clean_df(all_files):

    cleaned_files = []

    

    for file in tqdm(all_files):

        features = [

            file['paper_id'],

            file['metadata']['title'],

            format_authors(file['metadata']['authors']),

            format_authors(file['metadata']['authors'], 

                           with_affiliation=True),

            format_body(file['abstract']),

            format_body(file['body_text']),

            format_bib(file['bib_entries']),

            file['metadata']['authors'],

            file['bib_entries']

        ]



        cleaned_files.append(features)



    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    

    return clean_df
data_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

filenames = os.listdir(data_dir)

print("Number of articles retrieved from biorxiv:", len(filenames))
all_files = []



for filename in filenames:

    filename = data_dir + filename

    file = json.load(open(filename, 'rb'))

    all_files.append(file)
file = all_files[0]

print("Dictionary keys:", file.keys())
pprint(file['abstract'])
print("body_text type:", type(file['body_text']))

print("body_text length:", len(file['body_text']))

print("body_text keys:", file['body_text'][0].keys())
print("body_text content:")

pprint(file['body_text'][:10], depth=3)
texts = [(di['section'], di['text']) for di in file['body_text']]

texts_di = {di['section']: "" for di in file['body_text']}

for section, text in texts:

    texts_di[section] += text



pprint(list(texts_di.keys()))
body = ""



for section, text in texts_di.items():

    body += section

    body += "\n\n"

    body += text

    body += "\n\n"



print(body[:5000])
print(format_body(file['body_text'])[:1000])
print(all_files[0]['metadata'].keys())
print(all_files[0]['metadata']['title'])
authors = all_files[0]['metadata']['authors']

pprint(authors[:3])
for author in authors:

    print("Name:", format_name(author))

    print("Affiliation:", format_affiliation(author['affiliation']))

    print()
pprint(all_files[4]['metadata'], depth=4)
authors = all_files[4]['metadata']['authors']

print("Formatting without affiliation:")

print(format_authors(authors, with_affiliation=False))

print("\nFormatting with affiliation:")

print(format_authors(authors, with_affiliation=True))
bibs = list(file['bib_entries'].values())

pprint(bibs[:2], depth=4)
format_authors(bibs[1]['authors'], with_affiliation=False)
bib_formatted = format_bib(bibs[:5])

print(bib_formatted)
cleaned_files = []



for file in tqdm(all_files):

    features = [

        file['paper_id'],

        file['metadata']['title'],

        format_authors(file['metadata']['authors']),

        format_authors(file['metadata']['authors'], 

                       with_affiliation=True),

        format_body(file['abstract']),

        format_body(file['body_text']),

        format_bib(file['bib_entries']),

        file['metadata']['authors'],

        file['bib_entries']

    ]

    

    cleaned_files.append(features)
col_names = [

    'paper_id', 

    'title', 

    'authors',

    'affiliations', 

    'abstract', 

    'text', 

    'bibliography',

    'raw_authors',

    'raw_bibliography'

]



clean_df = pd.DataFrame(cleaned_files, columns=col_names)

clean_df.head()
clean_df.to_csv('biorxiv_clean.csv', index=False)
biorxiv_data = pd.read_csv('biorxiv_clean.csv')
biorxiv_data.head(2)
biorxiv_data.text[0]
import string
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

    

#punctuations = string.punctuation

#stop_words = list(STOPWORDS)



#parser = English()

def spacy_tokenizer(sentence):

    #reference and credits : https://www.kaggle.com/thebrownviking20/analyzing-quora-for-the-insinceres

    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]

    mytokens = " ".join([i for i in mytokens])

    return mytokens
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook



stopwords = set(STOPWORDS)

#https://www.kaggle.com/gpreda/cord-19-solution-toolbox



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=200,

        max_font_size=30, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(15,15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=14)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(biorxiv_data['abstract'], title = 'biorxiv_data - papers Abstract - frequent words (400 sample)')
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

stopwords = list(STOPWORDS)



#parser = English()

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
# Credit : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

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
plot_wordcloud(biorxiv_data['affiliations'].values, title="Word Cloud of Affiliations in biorxiv medrxiv Data ",color = 'white')
df1 = biorxiv_data['abstract'].dropna()



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



# Creating subplots

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,

                          subplot_titles=["Frequent words in biorxiv_data"])

fig.append_trace(trace0, 1, 1)



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







# Creating 1 plot

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,horizontal_spacing=0.25,

                          subplot_titles=["Frequent words in biorxiv_data"])

fig.append_trace(trace0, 1, 1)

#fig.append_trace(trace1, 1, 2)

#fig.append_trace(trace2, 2, 1)

#fig.append_trace(trace3, 2, 2)





fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots of Abstracts")

iplot(fig, filename='word-plots')
for sent in df1:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'blue')





freq_dict = defaultdict(int)











# Creating two subplots

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04, horizontal_spacing=0.05,

                          subplot_titles=["Frequent words in biorxiv_data"])

fig.append_trace(trace0, 1, 1)

#fig.append_trace(trace1, 2, 1)

#fig.append_trace(trace2, 3, 1)

#fig.append_trace(trace3, 4, 1)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")

iplot(fig, filename='word-plots')
df1 = biorxiv_data['text'].dropna()



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









# Creating two subplots

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,

                          subplot_titles=["Frequent words in biorxiv_data"])

fig.append_trace(trace0, 1, 1)







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





# Creating two subplots

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,horizontal_spacing=0.25,

                          subplot_titles=["Frequent words in biorxiv_data"])

fig.append_trace(trace0, 1, 1)







fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots of Texts in papers")

iplot(fig, filename='word-plots')
%%time

text_1 = biorxiv_data["abstract"].dropna().apply(spacy_tokenizer)



#count vectorization

vectorizer_1= CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')





text1_vectorized = vectorizer_1.fit_transform(text_1)

%%time

lda1 = LatentDirichletAllocation(n_components=5, max_iter=10, learning_method='online',verbose=True)





lda_1 = lda1.fit_transform(text1_vectorized)

def selected_topics(model, vectorizer, top_n=10):

    for idx, topic in enumerate(model.components_):

        print("Topic %d:" % (idx))

        print([(vectorizer.get_feature_names()[i], topic[i])

                        for i in topic.argsort()[:-top_n - 1:-1]]) 
print("LDA Model of Bioarvix data Abstracts:")

selected_topics(lda1, vectorizer_1)
tqdm.pandas()

fre_1 = np.array(biorxiv_data["abstract"].dropna().apply(textstat.flesch_reading_ease))

plot_readability(fre_1,"Flesch Reading Ease",20)

dcr_ = np.array(biorxiv_data["abstract"].dropna().apply(textstat.dale_chall_readability_score))

plot_readability(dcr_,"Dale-Chall Readability Score",1,['#C65D17'])
ari_ = np.array(biorxiv_data["abstract"].dropna().apply(textstat.coleman_liau_index))

plot_readability(ari_,"Automated Readability Index",10,['#2B2D42'])
cli_ = np.array(biorxiv_data["abstract"].dropna().apply(textstat.coleman_liau_index))

plot_readability(cli_,"The Coleman-Liau Index",10,['#e8434e'])
fog_ = np.array(biorxiv_data["abstract"].dropna().apply(textstat.gunning_fog))

plot_readability(fog_,"The Fog Scale (Gunning FOG Formula)",4,['#d98714'])
#lets make one dataframe.

all_data = biorxiv_data
text = all_data['text'][1]

print('Text before summarizations:')

prGreen(text)
print('Text after summarization:')

prYellow(summarize(text,word_count = 100))
import tensorflow as tf

import os

from tensorflow.python.keras.layers import Layer

from tensorflow.python.keras import backend as K





class AttentionLayer(Layer):

    """

    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).

    There are three sets of weights introduced W_a, U_a, and V_a

     """



    def __init__(self, **kwargs):

        super(AttentionLayer, self).__init__(**kwargs)



    def build(self, input_shape):

        assert isinstance(input_shape, list)

        # Create a trainable weight variable for this layer.



        self.W_a = self.add_weight(name='W_a',

                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),

                                   initializer='uniform',

                                   trainable=True)

        self.U_a = self.add_weight(name='U_a',

                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),

                                   initializer='uniform',

                                   trainable=True)

        self.V_a = self.add_weight(name='V_a',

                                   shape=tf.TensorShape((input_shape[0][2], 1)),

                                   initializer='uniform',

                                   trainable=True)



        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

        

        

    def call(self, inputs, verbose=False):

        """

        inputs: [encoder_output_sequence, decoder_output_sequence]

        """

        assert type(inputs) == list

        encoder_out_seq, decoder_out_seq = inputs

        if verbose:

            print('encoder_out_seq>', encoder_out_seq.shape)

            print('decoder_out_seq>', decoder_out_seq.shape)



        def energy_step(inputs, states):

            """ Step function for computing energy for a single decoder state """



            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))

            assert isinstance(states, list) or isinstance(states, tuple), assert_msg



            """ Some parameters required for shaping tensors"""

            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]

            de_hidden = inputs.shape[-1]



            """ Computing S.Wa where S=[s0, s1, ..., si]"""

            # <= batch_size*en_seq_len, latent_dim

            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))

            # <= batch_size*en_seq_len, latent_dim

            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))

            if verbose:

                print('wa.s>',W_a_dot_s.shape)



            """ Computing hj.Ua """

            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim

            if verbose:

                print('Ua.h>',U_a_dot_h.shape)

            

            

            """ tanh(S.Wa + hj.Ua) """

            # <= batch_size*en_seq_len, latent_dim

            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))

            if verbose:

                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)



            """ softmax(va.tanh(S.Wa + hj.Ua)) """

            # <= batch_size, en_seq_len

            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))

            # <= batch_size, en_seq_len

            e_i = K.softmax(e_i)



            if verbose:

                print('ei>', e_i.shape)



            return e_i, [e_i]

        def context_step(inputs, states):

            """ Step function for computing ci using ei """

            # <= batch_size, hidden_size

            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)

            if verbose:

                print('ci>', c_i.shape)

            return c_i, [c_i]



        def create_inital_state(inputs, hidden_size):

            # We are not using initial states, but need to pass something to K.rnn funciton

            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim

            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)

            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)

            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim

            return fake_state



        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])

        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim



        """ Computing energy outputs """

        # e_outputs => (batch_size, de_seq_len, en_seq_len)

        last_out, e_outputs, _ = K.rnn(

            energy_step, decoder_out_seq, [fake_state_e],

        )



        """ Computing context vectors """

        last_out, c_outputs, _ = K.rnn(

            context_step, e_outputs, [fake_state_c],

        )



        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):

        """ Outputs produced by the layer """

        return [

            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),

            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))

        ]

import re           

from bs4 import BeautifulSoup 

from keras.preprocessing.text import Tokenizer 

from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords   

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping

import warnings

pd.set_option("display.max_colwidth", 200)

warnings.filterwarnings("ignore")
all_data.head(2)
all_data.drop_duplicates(subset=['text'],inplace=True)  #dropping duplicates

all_data.dropna(axis=0,inplace=True)   #dropping na
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",



                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",



                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",



                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",



                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",



                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",



                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",



                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",



                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",



                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",



                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",



                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",



                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",



                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",



                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",



                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",



                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",



                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",



                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",



                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",



                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",



                           "you're": "you are", "you've": "you have"}
all_data['text'][:10]
stop_words = set(stopwords.words('english')) 

def text_cleaner(text):

    newString = text.lower()

    newString = BeautifulSoup(newString, "lxml").text

    newString = re.sub(r'\([^)]*\)', '', newString)

    newString = re.sub('"','', newString)

    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    

    newString = re.sub(r"'s\b","",newString)

    newString = re.sub("[^a-zA-Z]", " ", newString) 

    tokens = [w for w in newString.split() if not w in stop_words]

    long_words=[]

    for i in tokens:

        if len(i)>=3:                  #removing short word

            long_words.append(i)   

    return (" ".join(long_words)).strip()



cleaned_text = []

for t in all_data['text']:

    cleaned_text.append(text_cleaner(t))
all_data['cleaned_text']=cleaned_text

all_data.dropna(axis=0,inplace=True)
all_data['abstract'][:10]
def abstract_cleaner(text):

    newString = re.sub('"','', text)

    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    

    newString = re.sub(r"'s\b","",newString)

    newString = re.sub("[^a-zA-Z]", " ", newString)

    newString = newString.lower()

    tokens=newString.split()

    newString=''

    for i in tokens:

        if len(i)>1:                                 

            newString=newString+i+' '  

    return newString



#Call the above function

cleaned_abstract = []

for t in all_data['abstract']:

    cleaned_abstract.append(abstract_cleaner(t))



all_data['cleaned_text']=cleaned_text

all_data['cleaned_abstract']=cleaned_abstract

all_data['cleaned_abstract'].replace('', np.nan, inplace=True)

all_data.dropna(axis=0,inplace=True)
all_data['cleaned_abstract'] = all_data['cleaned_abstract'].apply(lambda x : '_START_ '+ x + ' _END_')
for i in range(2):

    print("Paper Text:",all_data['cleaned_text'][i])

    print("Paper Abstract:",all_data['cleaned_abstract'][i])

    print("\n")
import matplotlib.pyplot as plt

text_word_count = []

abstract_word_count = []



# populate the lists with sentence lengths

for i in all_data['cleaned_text']:

      text_word_count.append(len(i.split()))



for i in all_data['cleaned_abstract']:

      abstract_word_count.append(len(i.split()))



length_df = pd.DataFrame({'text':text_word_count, 'abstract':abstract_word_count})

length_df.hist(bins = 30)

plt.show()
max_len_text = 3000

max_len_abstract = 250
from sklearn.model_selection import train_test_split

x_tr,x_val,y_tr,y_val=train_test_split(all_data['cleaned_text'],all_data['cleaned_abstract'],test_size=0.1,random_state=0,shuffle=True) 

#prepare a tokenizer for reviews on training data

x_tokenizer = Tokenizer()

x_tokenizer.fit_on_texts(list(x_tr))



#convert text sequences into integer sequences

x_tr    =   x_tokenizer.texts_to_sequences(x_tr) 

x_val   =   x_tokenizer.texts_to_sequences(x_val)



#padding zero upto maximum length

x_tr    =   pad_sequences(x_tr,  maxlen=max_len_text, padding='post') 

x_val   =   pad_sequences(x_val, maxlen=max_len_text, padding='post')



x_voc_size   =  len(x_tokenizer.word_index) +1
#preparing a tokenizer for abstract on training data 

y_tokenizer = Tokenizer()

y_tokenizer.fit_on_texts(list(y_tr))



#convert summary sequences into integer sequences

y_tr    =   y_tokenizer.texts_to_sequences(y_tr) 

y_val   =   y_tokenizer.texts_to_sequences(y_val) 



#padding zero upto maximum length

y_tr    =   pad_sequences(y_tr, maxlen=max_len_abstract, padding='post')

y_val   =   pad_sequences(y_val, maxlen=max_len_abstract, padding='post')



y_voc_size  =   len(y_tokenizer.word_index) +1
from keras import backend as K 

K.clear_session() 

latent_dim = 100 



# Encoder 

encoder_inputs = Input(shape=(max_len_text,)) 

enc_emb = Embedding(x_voc_size, latent_dim,trainable=True)(encoder_inputs) 



#LSTM 1 

encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) 

encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb) 



#LSTM 2 

encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True) 

encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 



#LSTM 3 

encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True) 

encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 



# Set up the decoder. 

decoder_inputs = Input(shape=(None,)) 

dec_emb_layer = Embedding(y_voc_size, latent_dim,trainable=True) 

dec_emb = dec_emb_layer(decoder_inputs) 



#LSTM using encoder_states as initial state

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) 

decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c]) 



#Attention Layer

attn_layer = AttentionLayer(name='attention_layer') 

attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs]) 



# Concat attention output and decoder LSTM output 

decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer

decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax')) 

decoder_outputs = decoder_dense(decoder_concat_input) 



# Define the model

model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 

model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
#history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs= 5,callbacks=[es],batch_size= 32, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))
#!pip install  tensorflow-gpu==1.15.0

#!pip install bert-tensorflow

#!pip install tensorflow_hub