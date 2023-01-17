# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import json
count = 0

file_exts = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        count += 1

        file_ext = filename.split(".")[-1]

        file_exts.append(file_ext)



file_ext_set = set(file_exts)



print(f"Files: {count}")

print(f"Files extensions: {file_ext_set}\n\n=====================\nFiles extension count:\n=====================")

file_ext_list = list(file_ext_set)

for fe in file_ext_list:

    fe_count = file_exts.count(fe)

    print(f"{fe}: {fe_count}")
count = 0

for root, folders, filenames in os.walk('../input'):

    print(root, folders)
json_folder_path = "../input/CORD-19-research-challenge/custom_license/custom_license"

json_file_name = os.listdir(json_folder_path)[0]

json_path = os.path.join(json_folder_path, json_file_name)



with open(json_path) as json_file:

    json_data = json.load(json_file)
json_data_df = pd.io.json.json_normalize(json_data)

json_data_df.head()
print(f"Files in folder: {len(os.listdir(json_folder_path))}")
from tqdm import tqdm



# to process all files, uncomment the next line and comment the line below

# list_of_files = list(os.listdir(json_folder_path))

list_of_files = list(os.listdir(json_folder_path))[0:400]

pmc_custom_license_df = pd.DataFrame()



for file in tqdm(list_of_files):

    json_path = os.path.join(json_folder_path, file)

    with open(json_path) as json_file:

        json_data = json.load(json_file)

    json_data_df = pd.io.json.json_normalize(json_data)

    pmc_custom_license_df = pmc_custom_license_df.append(json_data_df)
pmc_custom_license_df.head()
pmc_custom_license_df['abstract_text'] = pmc_custom_license_df['abstract'].apply(lambda x: x[0]['text'] if x else "")
pd.set_option('display.max_colwidth', 500)

pmc_custom_license_df[['abstract', 'abstract_text']].head()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline 

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

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
show_wordcloud(pmc_custom_license_df['abstract_text'], title = 'PMC Custom license - papers abstract - frequent words (400 sample)')
show_wordcloud(pmc_custom_license_df['bib_entries.BIBREF0.title'], title = 'PMC Custom license - papers title - frequent words (400 sample)')
pmc_custom_license_df.loc[((pmc_custom_license_df['bib_entries.BIBREF0.venue']=="") | ((pmc_custom_license_df['bib_entries.BIBREF0.venue'].isna()))), 'bib_entries.BIBREF0.venue'] = "Not identified"
import seaborn as sns
def plot_count(feature, title, df, size=1, show_percents=False):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[0:20], palette='Set3')

    g.set_title("Number of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=10)

    if(show_percents):

        for p in ax.patches:

            height = p.get_height()

            ax.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}%'.format(100*height/total),

                    ha="center") 

    ax.set_xticklabels(ax.get_xticklabels());

    plt.show()  
plot_count('bib_entries.BIBREF0.venue', 'PMC Custom license - Top 20 Journals (400 sample)', pmc_custom_license_df, 3.5)
meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
meta.head()
meta.describe()
meta.info()
import en_core_web_sm

nlp = en_core_web_sm.load()
# nlp = spacy.load("en_core_sci_sm")

vector_dict = {}

for sha, abstract in tqdm(meta[["sha","abstract"]].values):

    if isinstance(abstract, str):

        vector_dict[sha] = nlp(abstract).vector
from sklearn.metrics.pairwise import cosine_similarity
keys = list(vector_dict.keys())

values = list(vector_dict.values())
cosine_sim_matrix = cosine_similarity(values, values)
n_sim_articles = 5

input_sha = "e3b40cc8e0e137c416b4a2273a4dca94ae8178cc"





sha_index = keys.index(input_sha)

sim_indexes = np.argsort(cosine_sim_matrix[sha_index])[::-1][1:n_sim_articles+1]

sim_shas = [keys[i] for i in sim_indexes]

meta_info = meta[meta.sha.isin(sim_shas)]
print("-------QUERY ABSTRACT-----")

print(meta[meta.sha == input_sha]["abstract"].values[0])
print(f"----TOP {n_sim_articles} SIMILAR ABSTRACTS-----")

for abst in meta_info.abstract.values:

    print(abst)

    print("---------")
n_return = 5

query_statement = "Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies"
query_vector = nlp(query_statement).vector

cosine_sim_matrix_query = cosine_similarity(values, query_vector.reshape(1,-1))

query_sim_indexes = np.argsort(cosine_sim_matrix_query.reshape(1,-1)[0])[::-1][:n_return]

query_shas = [keys[i] for i in query_sim_indexes]

meta_info_query = meta[meta.sha.isin(query_shas)]
print(f"----TOP {n_return} SIMILAR ABSTRACTS TO QUERY-----")

for abst in meta_info_query.abstract.values:

    print(abst)

    print("---------")
import numpy as np

import gensim

import os

import re



from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from gensim import corpora



from gensim.models.ldamulticore import LdaMulticore



import pandas as pd
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz
# df = pd.read_csv('metadata.csv')

bucket = 'coviddata'

file = 'metadata.csv'

gcs_url = 'https://%(bucket)s.storage.googleapis.com/%(file)s' % {'bucket':bucket, 'file':file}

df = pd.read_csv(gcs_url)
df.head()
df2 = df.drop(columns = ['sha', 'source_x', 'pmcid', 'license', 'Microsoft Academic Paper ID', 'WHO #Covidence', 'has_full_text'])
df2.head()
df2.shape
df3 = df2.dropna(subset=['abstract'])

df3.shape
df3.head()

import en_core_sci_md

nlp = en_core_sci_md.load(disable=["tagger", "parser", "ner"])

nlp.max_length = 2000000
import spacy
from spacy.tokenizer import Tokenizer

def tokenize(doc):

    

    return [token.text for token in nlp(doc) if not token.is_stop and not token.is_punct and not token.pos == 'PRON']
data = df3['abstract'].apply(tokenize)

data
vect = [nlp(doc).vector for doc in df3['abstract']]
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=25, algorithm='ball_tree')

nn.fit(vect)
query = "chloroquine hydroxycholoroquine HCoV-19 SARS-CoV-2 coronavirus covid-19 treatment"
query_vect = nlp(query).vector


similar_abstracts = nn.kneighbors([query_vect])[1]
for abstract in similar_abstracts:

    print(df3['abstract'].iloc[abstract])
output = pd.DataFrame((df3['abstract'].iloc[abstract]))

pd.set_option('display.max_colwidth', 0)

output.head(25)
query1 = output.iloc[ 10, : ]

query1.head()
query2 = output.iloc[ 19, : ]

query2.head()
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words='english', tokenizer = tokenize, ngram_range=(1,2))

tf = vect.fit_transform(output['abstract'])
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=50, random_state=0, n_jobs=-1)

lda.fit(tf)
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "\nTopic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()
tfidf_feature_names = vect.get_feature_names()

top_words = print_top_words(lda, tfidf_feature_names, 25)

top_words
!pip install pyLDAvis
import pyLDAvis.gensim



pyLDAvis.enable_notebook()
data = output['abstract'].apply(tokenize)

id2word = corpora.Dictionary(data)

corpus = [id2word.doc2bow(token) for token in data]

lda2 = LdaMulticore(corpus = corpus,

                   id2word = id2word,

                   random_state = 42,

                   num_topics = 15,

                   passes = 10,

                   workers = 4)
lda2.print_topics()
import re

words = [re.findall(r'"([^"]*)"',t[1]) for t in lda2.print_topics()]
topics = [' '.join(t[0:10]) for t in words]
for id, t in enumerate(topics): 

    print(f"------ Topic {id} ------")

    print(t, end="\n\n")
pyLDAvis.gensim.prepare(lda2, corpus, id2word)