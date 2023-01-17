# !pip install pyecharts
# !pip install Pillow

# !pip install numpy torchvision_nightly
# !pip install covid19_tools

# !pip install spacy # Uncomment this if spacy package is not installed.
# !pip uninstall spacy # Uncomment this if installed version of spacy fails.
# !python -m spacy download en # Uncomment this if en language is not loaded in spacy package. 

# !pip install bert-tensorflow
# !pip install  tensorflow-gpu==1.15.0

!pip install spacy-langdetect
!pip install language-detector
!pip install symspellpy
!pip install sentence-transformers
import os
import re

import json
import math
import glob
import time

import string
import random
import pickle

import functools
import collections

from tqdm import tqdm
from PIL import Image
import seaborn as sns

from nltk import PorterStemmer
import torch.nn.functional as F


# from pyecharts.charts import Graph
# from pyecharts import options as opts
from keras.preprocessing import sequence
from scipy.spatial.distance import cdist
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from IPython.core.display import display, HTML

from nltk.tokenize import word_tokenize, sent_tokenize
from torch.utils.data import Dataset,TensorDataset,DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
# check version 
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

import tensorflow as tf
print("tensorflow version: {}". format(tf.__version__))

import torch #collection of machine learning algorithms
print("torch version: {}". format(torch.__version__))


# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
print('-'*71)


from subprocess import check_output
print('input file is:', check_output(["ls", "../input"]).decode("utf8"))


# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# detect and init the TPU

# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
# with tpu_strategy.scope():
#     model = tf.keras.Sequential( … ) # define your model normally
#     model.compile( … )

# # train model normally
# model.fit(training_dataset, epochs=EPOCHS, steps_per_epoch=…)



## try to use the tpu of Kaggle
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
from kaggle_datasets import KaggleDatasets
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.
print("REPLICAS: ", strategy.num_replicas_in_sync)
!ls /kaggle/input/CORD-19-research-challenge/  # the content of input files
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'  
with open('../input/CORD-19-research-challenge/metadata.readme', 'r') as f:
    data = f.read()
    print(data)
## load the meta data from the CSV file 
df = pd.read_csv(metadata_path,header = 0,  # you can specify the header where it is from
                 usecols=['title','abstract','authors','doi','publish_time','source_x'],   # select use columns
                 dtype={
                        'Microsoft Academic Paper ID': str,
                        'pubmed_id': str,
                        'doi': str,                       
                       },
                 low_memory=False)

print (f'The shape of the input data:\n{df.shape[0]} articles, every article has {df.shape[1]} features')
print(df.info())
print(f'None data:\n{df.isnull().sum()}')
df.sample(3)

# before 2020, maybe publication about COVID-2019 is not out.
df_2020 = df.query("'2020' in publish_time")

# I think Confidence Interval (CI) is used epidemiological evaluation
df_2020_ci = df_2020.loc[df_2020["abstract"].str.contains("CI").fillna(False), :]

print(df_2020_ci.shape)
df_2020_ci.head()
biorxiv_dir = '../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))
from wordcloud import WordCloud, STOPWORDS
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

    fig = plt.figure(1, figsize=(20,5))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    
    
show_wordcloud(df['abstract'], title = 'metadata - papers Abstract - frequent words (400 sample)')

## Convert abstract to list
# data = df.abstract.dropna().values.tolist()
### Fetch All of JSON File Path
# all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
# len(all_json)
# print(all_json)

# # Checking JSON Schema Structure
# with open(all_json[0]) as file:
#     first_entry = json.load(file)
#     print(json.dumps(first_entry, indent=4))
    
    
# load the json file in the directory
dirs_=["/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json",
"/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json",
"/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json",
"/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json"]
 

data_ = list()
for dir_ in dirs_:
	for filename in tqdm(os.listdir(dir_)):

		x=str(dir_)+'/'+str(filename)
        
		with open(x) as file:
			data=json.loads(file.read())
		
		#take out the data from the json format
		paper_id=data['paper_id']
		meta_data=data['metadata']
		abstract=data['abstract']
		abstract_text=""
		for text in abstract:
			abstract_text+=text['text']+" "
		body_text=data['body_text']
		full_text=""
		for text in body_text:
			full_text+=text['text']+" "
		back_matter=data['back_matter']
		#store everything to a dataframe
		data_.append([paper_id,abstract_text,full_text])

df_json = pd.DataFrame(data_,columns=['paper_id','abstract','full_text'])
print(df.head())
#save as a csv
#df.to_csv('biorxiv_medrxiv.csv', index = True)
df_json.to_csv('train.csv', index = True)
#a data frame for my complete body.
df_json.head()
df_copy = df.copy() #remember python assignment or equal passes by reference vs values, so use copy()
# data_cleaner = [data1, df]  # however passing by reference is convenient, because we can clean both datasets at once

# df = df.drop_duplicates(subset='abstract', keep="first")  # df=df.drop_duplicates() # df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
# df_covid.describe(include='all')

df = df.dropna()  # drop NANs 

df["abstract"] = df["abstract"].str.lower()  # convert abstracts to lowercase

print (f'after preproccess: the shape :{df.shape}')   # print (f'head input data information: {df.head()}')   # show 5 lines of the new dataframe

df.info()
df.head()
sourceDic = collections.defaultdict(int)
for s in df["source_x"][df["source_x"].notnull()]:
    sourceDic[s] += 1
sizes, explode, labels = [], [], []
for s in sourceDic:
    sizes.append(sourceDic[s])
    explode.append(0)
    labels.append(s)
    
colors = ['red', 'gold', 'lightcoral', 'violet', 'lightskyblue', 'green']
fig = plt.gcf()
fig.set_size_inches(8, 8)
plt.pie(sizes, explode=explode, labels=labels, colors = colors, autopct='%1.2f%%', shadow=True, startangle=140)
plt.title('Literature source distribution')
plt.axis('equal')
plt.show()
yearList = []
for y in df["publish_time"][df["publish_time"].notnull()]:
    yearList.append(int(re.split(' |-', y)[0]))

sns.distplot(yearList, bins = 50)
plt.title("Publish year distribution")
plt.xlabel("Year")
plt.ylabel("Frequency")
print("The number of articles with abstract: " + str(sum(df["abstract"].notnull())))
from nltk.corpus import stopwords 
from nltk.corpus import wordnet

startTime = time.time()
absLength = []
word2count = {}
for abstract in df["abstract"][df["abstract"].notnull()]:
    ## Remove web links
    abstract = re.sub('https?://\S+|www\.\S+', '', abstract) 

    ## Lowercase
    abstract = abstract.lower()
    
    ## Remove punctuation
    abstract = re.sub('<.*?>+', ' ', abstract)
    abstract = re.sub('[%s]' % re.escape(string.punctuation), ' ', abstract)
    
    ## Remove number
    abstract = re.sub(r'\d+', '', abstract)
    
    ## Tokenize
    words = word_tokenize(abstract)
    
    ## Remove stop words
    nltk_stop_words = stopwords.words('english')
    words = [word for word in words if word not in nltk_stop_words]
    
    ## Stem
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]
    
    ## Lematize verbs
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    
    ## Record length
    absLength.append(len(words))
    
    ## Get word count
    for word in words:
        count = word2count.get(word, 0)
        word2count[word] = count + 1
print("Time spent: " + str(round((time.time() - startTime) / 60, 3)) + "min.")
print("The number of tokens: " + str(len(word2count)))
sns.distplot(sorted(absLength)[:-20], bins = 50) # There are 20 extremely long abstracts
plt.xlabel("Abstract token count")
plt.ylabel("Frequency")
plt.show()
df_word_count = pd.DataFrame(sorted(word2count.items(), key=lambda x: x[1])[::-1])
sns.set(rc={'figure.figsize':(12,10)})
sns.barplot(y = df_word_count[0].values[:50], x = df_word_count[1].values[:50], color='red')
def textNormalize(rawString):
    """
    Function for text normalization.
    Text normalization includes:
    1. removing web links
    2. converting all letters to lower or upper case
    3. removing punctuationsz
    4. removing numbers
    5. tokenization
    6. removing stopwords
    7. stemming
    8. lemmatization
    Input:
        rawString: a string contains the text to be normaized. 
    Output:
        normText: a string contains the normalized text where the tokens extracted from rawString are joined by space.
    """
    if rawString == np.nan:
        return rawString
    ## Remove web links
    rawString = re.sub('https?://\S+|www\.\S+', '', rawString) 

    ## Lowercase
    rawString = rawString.lower()
    
    ## Remove punctuation
    rawString = re.sub('<.*?>+', ' ', rawString)
    rawString = re.sub('[%s]' % re.escape(string.punctuation), ' ', rawString)
    
    ## Remove number
    rawString = re.sub(r'\d+', '', rawString)
    
    ## Tokenize
    words = word_tokenize(rawString)
    
    ## Remove stop words
    nltk_stop_words = stopwords.words('english')
    words = [word for word in words if word not in nltk_stop_words]
    
    ## Stem
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]
    
    ## Lematize verbs
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    
    normText = " ".join(words)
    
    return normText


startTime = time.time()
df["clean_abstract"] = float("NaN")
df.loc[df["abstract"].notnull(), "clean_abstract"] = \
df["abstract"][df["abstract"].notnull()].apply(lambda x: textNormalize(x))
print("Time spent: " + str(round((time.time() - startTime) / 60, 3)) + "min.")

df.to_csv("Metadata_clean.csv", index = False)
### Utils
from collections import Counter
from sklearn.metrics import silhouette_score
import umap
from wordcloud import WordCloud
from gensim.models.coherencemodel import CoherenceModel


def get_topic_words(token_lists, labels, k=None):
    """
    get top words within each topic from clustering results
    """
    if k is None:
        k = len(np.unique(labels))
    topics = ['' for _ in range(k)]
    for i, c in enumerate(token_lists):
        topics[labels[i]] += (' ' + ' '.join(c))
    word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
    # get sorted word counts
    word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts))
    # get topics
    topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))

    return topics

def get_coherence(model, token_lists, measure='c_v'):
    """
    Get model coherence from gensim.models.coherencemodel
    :param model: Topic_Model object
    :param token_lists: token lists of docs
    :param topics: topics as top words
    :param measure: coherence metrics
    :return: coherence score
    """
    if model.method == 'LDA':
        cm = CoherenceModel(model=model.ldamodel, texts=token_lists, corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    else:
        topics = get_topic_words(token_lists, model.cluster_model.labels_)
        cm = CoherenceModel(topics=topics, texts=token_lists, corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    return cm.get_coherence()

def get_silhouette(model):
    """
    Get silhouette score from model
    :param model: Topic_Model object
    :return: silhouette score
    """
    if model.method == 'LDA':
        return
    lbs = model.cluster_model.labels_
    vec = model.vec[model.method]
    return silhouette_score(vec, lbs)

def plot_proj(embedding, lbs):
    """
    Plot UMAP embeddings
    :param embedding: UMAP (or other) embeddings
    :param lbs: labels
    """
    n = len(embedding)
    counter = Counter(lbs)
    for i in range(len(np.unique(lbs))):
        plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', alpha=0.5,
                 label='cluster {}: {:.2f}%'.format(i, counter[i] / n * 100))
    plt.legend(loc = 'best')
    plt.grid(color ='grey', linestyle='-',linewidth = 0.25)


def visualize(model):
    """
    Visualize the result for the topic model by 2D embedding (UMAP)
    :param model: Topic_Model object
    """
    if model.method == 'LDA':
        return
    reducer = umap.UMAP()
    print('Calculating UMAP projection ...')
    vec_umap = reducer.fit_transform(model.vec[model.method])
    print('Calculating UMAP projection. Done!')
    plot_proj(vec_umap, model.cluster_model.labels_)
    dr = '/kaggle/working/contextual_topic_identification/docs/images/{}/{}'.format(model.method, model.id)
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig('/kaggle/working/2D_vis')

def get_wordcloud(model, token_lists, topic):
    """
    Get word cloud of each topic from fitted model
    :param model: Topic_Model object
    :param sentences: preprocessed sentences from docs
    """
    if model.method == 'LDA':
        return
    print('Getting wordcloud for topic {} ...'.format(topic))
    lbs = model.cluster_model.labels_
    tokens = ' '.join([' '.join(_) for _ in np.array(token_lists)[lbs == topic]])

    wordcloud = WordCloud(width=800, height=560,
                          background_color='white', collocations=False,
                          min_font_size=10).generate(tokens)

    # plot the WordCloud image
    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    dr = '/kaggle/working/{}/{}'.format(model.method, model.id)
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig('/kaggle/working' + '/Topic' + str(topic) + '_wordcloud')
    print('Getting wordcloud for topic {}. Done!'.format(topic))
### Preprocessing 

from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
from language_detector import detect_language

import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


###################################
#### sentence level preprocess ####
###################################

# lowercase + base filter, some basic normalization
def f_base(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: "&gt", "&lt"
    s = re.sub(r'&gt|&lt', ' ', s)
    # normalization 4: letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # normalization 5: non-word repetition (if more than 1)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)
    # normalization 6: string * as delimiter
    s = re.sub(r'\*|\W\*|\*\W', '. ', s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    s = re.sub(r'\(.*?\)', '. ', s)
    # normalization 8: xxx[?!]. -- > xxx.
    s = re.sub(r'\W+?\.', '.', s)
    # normalization 9: [.?!] --> [.?!] xxx
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    # normalization 10: ' ing ', noise text
    s = re.sub(r' ing ', ' ', s)
    # normalization 11: noise text
    s = re.sub(r'product received for free[.| ]', ' ', s)
    # normalization 12: phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)

    return s.strip()


# language detection
def f_lan(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """
    
    # some reviews are actually english but biased toward french
    return detect_language(s) in {'English', 'French','Spanish','Chinese'}


###############################
#### word level preprocess ####
###############################

# filtering out punctuations and numbers
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word.isalpha()]


# selecting nouns
def f_noun(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with only nouns selected
    """
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']


# typo correction
def f_typo(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with typo fixed by symspell. words with no match up will be dropped
    """
    w_list_fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            w_list_fixed.append(suggestions[0].term)
        else:
            pass
            # do word segmentation, deprecated for inefficiency
            # w_seg = sym_spell.word_segmentation(phrase=word)
            # w_list_fixed.extend(w_seg.corrected_string.split())
    return w_list_fixed


# stemming if doing word-wise
p_stemmer = PorterStemmer()


def f_stem(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    return [p_stemmer.stem(word) for word in w_list]


# filtering out stop words
# create English stop words list

stop_words = (list(
    set(get_stop_words('en'))
    |set(get_stop_words('es'))
    |set(get_stop_words('de'))
    |set(get_stop_words('it'))
    |set(get_stop_words('ca'))
    #|set(get_stop_words('cy'))
    |set(get_stop_words('pt'))
    #|set(get_stop_words('tl'))
    |set(get_stop_words('pl'))
    #|set(get_stop_words('et'))
    |set(get_stop_words('da'))
    |set(get_stop_words('ru'))
    #|set(get_stop_words('so'))
    |set(get_stop_words('sv'))
    |set(get_stop_words('sk'))
    #|set(get_stop_words('cs'))
    |set(get_stop_words('nl'))
    #|set(get_stop_words('sl'))
    #|set(get_stop_words('no'))
    #|set(get_stop_words('zh-cn'))
))




def f_stopw(w_list):
    """
    filtering out stop words
    """
    return [word for word in w_list if word not in stop_words]


def preprocess_sent(rw):
    """
    Get sentence level preprocessed data from raw review texts
    :param rw: review to be processed
    :return: sentence level pre-processed review
    """
    s = f_base(rw)
    if not f_lan(s):
        return None
    return s


def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, fix typo, stem, stop_words
    :param s: sentence to be processed
    :return: word level pre-processed review
    """
    if not s:
        return None
    w_list = word_tokenize(s)
    w_list = f_punct(w_list)
    w_list = f_noun(w_list)
    w_list = f_typo(w_list)
    w_list = f_stem(w_list)
    w_list = f_stopw(w_list)

    return w_list
### Autoencoder
import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class Autoencoder:
    """
    Autoencoder for learning latent space representation
    architecture simplified for only one hidden layer
    """

    def __init__(self, latent_dim=32, activation='relu', epochs=200, batch_size=128):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        """
        compile the computational graph
        """
        input_vec = Input(shape=(input_dim,))
        encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = Dense(input_dim, activation=self.activation)(encoded)
        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)
        encoded_input = Input(shape=(self.latent_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
        self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

    def fit(self, X):
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X)
        self.his = self.autoencoder.fit(X_train, X_train,
                                        epochs=200,
                                        batch_size=128,
                                        shuffle=True,
                                        validation_data=(X_test, X_test), verbose=0)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
#from Autoencoder import *
#from preprocess import *
from datetime import datetime


def preprocess(docs, samp_size=None):
    """
    Preprocess the data
    """
    if not samp_size:
        samp_size = 100

    print('Preprocessing raw texts ...')
    n_docs = len(docs)
    sentences = []  # sentence level preprocessed
    token_lists = []  # word level preprocessed
    idx_in = []  # index of sample selected
    #     samp = list(range(100))
    samp = np.random.choice(n_docs, samp_size)
    for i, idx in enumerate(samp):
        sentence = preprocess_sent(docs[idx])
        token_list = preprocess_word(sentence)
        if token_list:
            idx_in.append(idx)
            sentences.append(sentence)
            token_lists.append(token_list)
        print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')
    print('Preprocessing raw texts. Done!')
    return sentences, token_lists, idx_in


# define model object
class Topic_Model:
    def __init__(self, k=10, method='TFIDF'):
        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        if method not in {'TFIDF', 'LDA', 'BERT', 'LDA_BERT'}:
            raise Exception('Invalid method!')
        self.k = k
        self.dictionary = None
        self.corpus = None
        #         self.stopwords = None
        self.cluster_model = None
        self.ldamodel = None
        self.vec = {}
        self.gamma = 15  # parameter for reletive importance of lda
        self.method = method
        self.AE = None
        self.id = method + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def vectorize(self, sentences, token_lists, method=None):
        """
        Get vecotr representations from selected methods
        """
        # Default method
        if method is None:
            method = self.method

        # turn tokenized documents into a id <-> term dictionary
        self.dictionary = corpora.Dictionary(token_lists)
        # convert tokenized documents into a document-term matrix
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        if method == 'TFIDF':
            print('Getting vector representations for TF-IDF ...')
            tfidf = TfidfVectorizer()
            vec = tfidf.fit_transform(sentences)
            print('Getting vector representations for TF-IDF. Done!')
            return vec

        elif method == 'LDA':
            print('Getting vector representations for LDA ...')
            if not self.ldamodel:
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)

            def get_vec_lda(model, corpus, k):
                """
                Get the LDA vector representation (probabilistic topic assignments for all documents)
                :return: vec_lda with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_lda = np.zeros((n_doc, k))
                for i in range(n_doc):
                    # get the distribution for the i-th document in corpus
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vec_lda[i, topic] = prob

                return vec_lda

            vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
            print('Getting vector representations for LDA. Done!')
            return vec

        elif method == 'BERT':

            print('Getting vector representations for BERT ...')
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('bert-base-nli-max-tokens')
            vec = np.array(model.encode(sentences, show_progress_bar=True))
            print('Getting vector representations for BERT. Done!')
            return vec

             
        elif method == 'LDA_BERT':
        #else:
            vec_lda = self.vectorize(sentences, token_lists, method='LDA')
            vec_bert = self.vectorize(sentences, token_lists, method='BERT')
            vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
            self.vec['LDA_BERT_FULL'] = vec_ldabert
            if not self.AE:
                self.AE = Autoencoder()
                print('Fitting Autoencoder ...')
                self.AE.fit(vec_ldabert)
                print('Fitting Autoencoder Done!')
            vec = self.AE.encoder.predict(vec_ldabert)
            return vec

    def fit(self, sentences, token_lists, method=None, m_clustering=None):
        """
        Fit the topic model for selected method given the preprocessed data
        :docs: list of documents, each doc is preprocessed as tokens
        :return:
        """
        # Default method
        if method is None:
            method = self.method
        # Default clustering method
        if m_clustering is None:
            m_clustering = KMeans

        # turn tokenized documents into a id <-> term dictionary
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(token_lists)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        ####################################################
        #### Getting ldamodel or vector representations ####
        ####################################################

        if method == 'LDA':
            if not self.ldamodel:
                print('Fitting LDA ...')
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)
                print('Fitting LDA Done!')
        else:
            print('Clustering embeddings ...')
            self.cluster_model = m_clustering(self.k)
            self.vec[method] = self.vectorize(sentences, token_lists, method)
            self.cluster_model.fit(self.vec[method])
            print('Clustering embeddings. Done!')

    def predict(self, sentences, token_lists, out_of_sample=None):
        """
        Predict topics for new_documents
        """
        # Default as False
        out_of_sample = out_of_sample is not None

        if out_of_sample:
            corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            if self.method != 'LDA':
                vec = self.vectorize(sentences, token_lists)
                print(vec)
        else:
            corpus = self.corpus
            vec = self.vec.get(self.method, None)

        if self.method == "LDA":
            lbs = np.array(list(map(lambda x: sorted(self.ldamodel.get_document_topics(x),
                                                     key=lambda x: x[1], reverse=True)[0][0],
                                    corpus)))
        else:
            lbs = self.cluster_model.predict(vec)
        return lbs
### Training
#from model import *
#from utils import *
import pickle
import argparse
warnings.filterwarnings('ignore', category=Warning)  # import warnings


def main():  #def model(): #:if __name__ == '__main__':  
    
    method = "BERT"
    samp_size = 50000
    ntopic = 20
    
    #parser = argparse.ArgumentParser(description='contextual_topic_identification tm_test:1.0')
    #parser.add_argument('--fpath', default='/kaggle/working/train.csv')
    #parser.add_argument('--ntopic', default=10,)
    #parser.add_argument('--method', default='TFIDF')
    #parser.add_argument('--samp_size', default=20500)
    
    #args = parser.parse_args()

    data = pd.read_csv('/kaggle/working/train.csv')
    data = data.fillna('')  # only the comments has NaN's
    rws = data.abstract
    sentences, token_lists, idx_in = preprocess(rws, samp_size=samp_size)
    # Define the topic model object
    #tm = Topic_Model(k = 10), method = TFIDF)
    tm = Topic_Model(k = ntopic, method = method)
    # Fit the topic model by chosen method
    tm.fit(sentences, token_lists)
    # Evaluate using metrics
    with open("/kaggle/working/{}.file".format(tm.id), "wb") as f:
        pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)

    print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
    print('Silhouette Score:', get_silhouette(tm))
    # visualize and save img
    visualize(tm)
    for i in range(tm.k):
        get_wordcloud(tm, token_lists, i)

    
main()  # the model training now need to take more than 6 hours

### TODO:

# * Implement models.ldamulticore – parallelized Latent Dirichlet Allocation using all CPU cores to parallelize and speed up model training.
# * Switch from BERT/RoBERTa to SciBERT, BART, and or other models. 
# class Bandit:  # Reference : https://www.wikiwand.com/en/Algorithms_for_calculating_variance
#   def __init__(self, m):
#     """
#     :param m  : the real mean reward of the bandit. (not visible to the algorithm)
#     """
#     self.m = m   # the real mean reward
#     self.mean = 0  
#     self.var = float('inf')  # naive algorithm to calculate the estimated variance 
#     self.N = 0  # arm_count
#     self.sum = 0
#     self.sumSq = 0
#     # self.alpha = 0.5  # win weight/times
#     # self.beta = 0.5  # lose weight

#   def pull(self):
#     """Generate the random numbers
#         observes the reward. The reward should be the real mean reward 
#         +/- a random value 
#     """
#     if self. N < 0:
#       return False
#     p = np.random.randn(5)
#     # reward = self.m + np.random.normal(self.N, 1)
#     reward = self.m + p[0] * 10  # larger variance
#     return reward

#   def update(self, x):
#     """ update the stats
#     """
#     self.N += 1
#     self.sum += x
#     self.sumSq += x * x
#     self.mean = self.sum / self.N
#     self.var = (self.sumSq - self.sum * self.sum / self.N) / (self.N - 1)

#     # self.alpha +=  self.mean
#     # self.beta +=  (1 - self.mean)

#     return x



# def calculate_delta(T, item, chosen_count):
#     return 1 if chosen_count[item] == 0 else np.sqrt(2 * np.log(T) / chosen_count[item])
    
# def ucb(bandits, n):
#   """ select the bandit to pull using upper confident bound
#   :param bandits - the list of bandit objects to choose from
#   :param n - the number of repeated experiments
#   :return j - the index of the bandit to pull
#   """
#   d = len(bandits)
#   estimated_rewards = []  # estimated rewards of each bandit
#   chosen_count = np.zeros(d) 

#   for i in bandits:
#     estimated_rewards.append(i.mean)

#   upper_bound_probs = [estimated_rewards[item] + calculate_delta(n, item, chosen_count) for item in range(d)]
        
#   return np.argmax(upper_bound_probs)


# def epsilon_greedy(bandits, eps):
#   """ select the bandit to pull using epsilon greedy
#   :param bandits - the list of bandit objects to choose from
#   :param eps - epsilon probability of random action 0 < eps < 1 (float)
#   :return j - the index of the bandit to pull
#   """
#   # Mean reward for each arm
#   k_reward = []
#   for i in bandits:
#     k_reward.append(i.m)

#   p = np.random.rand()

#   if p < eps:
#     j = np.random.choice(len(bandits))
#   else:  # greedy action
#     j = np.argmax(k_reward) # pick up the max meal reward for each bandit
#   return j


# import  pymc

# def thompson_sampling(bandits):
#   """ select the bandit to pull using Thompson Sampling
#   :param bandits - the list of bandit objects to choose from
#   :param n - the number of repeated experiments
#   :return j - the index of the bandit to pull
#   """
#   thetas = []
#   for i in bandits:
#     thetas.append(random.normalvariate(i.mean, math.sqrt(i.var)))
#   return   np.argmax(thetas) 


# def run_experiment(bandits, eps, N, strategy='epsilon_greedy'):
#   data = np.empty(N)
  
#   for i in range(N):
#     # epsilon greedy
#     if strategy == 'epsilon_greedy':
#       j = epsilon_greedy(bandits, eps)
#     elif strategy == 'ucb':
#       j = ucb(bandits, i)
#     elif strategy == 'thompson_sampling':
#       j = thompson_sampling(bandits)

#     else:
#       j = np.random.choice(len(bandits))
#     x = bandits[j].pull()
#     bandits[j].update(x)

#     # for the plot
#     data[i] = x
#   cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
#   return cumulative_average


# # a =  [Bandit(1.1), Bandit(1.2), Bandit(1.3)]
# # bandits = [Bandit(1), Bandit(3), Bandit(5)]
# # c_0 = run_experiment(bandits, 0.0, 1000)

# # bandits = [Bandit(1), Bandit(3), Bandit(5)]
# # c_1 = run_experiment(bandits, 0.1, 1000)

# # bandits = [Bandit(1), Bandit(3), Bandit(5)]
# # c_05 = run_experiment(bandits, 0.05, 1000)

# # bandits = [Bandit(1), Bandit(3), Bandit(5)]
# # c_01 = run_experiment(bandits, 0.01, 1000)

# # bandits = [Bandit(1), Bandit(3), Bandit(5)]
# # c_1 = run_experiment(bandits, 1, 1000)

# # plt.plot(c_0, label='eps = 0')
# # plt.plot(c_1, label='eps = 0.1')
# # plt.plot(c_05, label='eps = 0.05')
# # plt.plot(c_01, label='eps = 0.01')
# # plt.plot(c_1, label='eps = 1')
# # plt.legend()
# # plt.show()


# # bandits = [Bandit(1.1), Bandit(1.2), Bandit(1.3),Bandit(1.4), Bandit(1.5), Bandit(1.6), Bandit(1.7), Bandit(1.8), Bandit(1.9)]
# # c_eps = run_experiment(bandits, 0.05, 1000, strategy='epsilon_greedy')

# # bandits = [Bandit(1.1), Bandit(1.2), Bandit(1.3),Bandit(1.4), Bandit(1.5), Bandit(1.6), Bandit(1.7), Bandit(1.8), Bandit(1.9)]
# # c_ucb = run_experiment(bandits, 0.05, 1000, strategy='ucb')

# # bandits = [Bandit(1.1), Bandit(1.2), Bandit(1.3),Bandit(1.4), Bandit(1.5), Bandit(1.6), Bandit(1.7), Bandit(1.8), Bandit(1.9)]
# # c_tp = run_experiment(bandits, 0.05, 1000, strategy='thompson_sampling')

# # plt.plot(c_eps, label='eps = 0.05')
# # plt.plot(c_ucb, label='ucb')
# # plt.plot(c_tp, label='thompson sampling')
# # plt.legend()
# # plt.show()
# Manual list for highlighting
# Need to look at automated / updateable approach to identifying these
risk_factors = [
    'diabetes',
    'hypertension',
    'smoking',
    'cardiovascular disease',
    'chronic obstructive pulmonary disease',
    'cerebrovascular disease',
    'kidney disease',
    ' age ',
    ' aged',
    'blood type',
    'hepatitis',
    ' male ',
    ' female ',
    ' males ',
    ' females ',
    'arrhythmia',
    ' sex ',
    ' gender ',
    'acute respiratory distress syndrome',
    'sepsis shock',
    'cardiac injury',
    'acute kidney injury',
    'liver dysfunction',
    'gastrointestinal haemorrhage',
    'conjunctivitis',
    'comorbidity',
    'comorbidities',
    'co-morbidity',
    'co-morbidities',
    ' smoker',
    'non-smoker'
]

# Reference: https://www.kaggle.com/maksimeren/covid-19-literature-clustering#Unsupervised-Learning:-Clustering-with-K-Means
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import HashingVectorizer

# hash vectorizer instance
hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)
n_gram_all = []

dict = pd.read_csv('/kaggle/working/train.csv').fillna('')  # only the comments has NaN's
dictlist = key = value = []
for key, value in dict.items():
    temp = [key,value]
    dictlist.append(temp)
    
for word in key:
    # get n-grams for the instance
    n_gram = []
    for i in range(len(word)-2+1):
        n_gram.append("".join(word[i:i+2]))
    n_gram_all.append(n_gram)
    
# features matrix X
X = hvec.fit_transform(n_gram_all)

from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print("X_test size:", len(X_test), "\n")

k = 7 
kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)
y_pred = kmeans.fit_predict(X_train)

# add labels
y_train = y_pred  # Labels for the training set:
y_test = kmeans.predict(X_test)  # Labels for the test set:
from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=5)   # Dimensionality Reduction with t-SNE
X_embedded = tsne.fit_transform(X_train)

# plot the t-SNE. scatterplot again and see if we have any obvious clusters after we have labels
# sns settings 
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("t-SNE Covid-19 Articles - Clustered")
# plt.savefig("plots/t-sne_covid19_label.png")
plt.show()
# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("t-SNE Covid-19 Articles - Clustered")
# plt.savefig("plots/t-sne_covid19_label.png")
plt.show()
# export to .csv file
# risk_factors_df = pd.DataFrame(csv_data)
# risk_factors_df.to_csv('risk_factors.csv', index=False)
# risk_factors_df.head()