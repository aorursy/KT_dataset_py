# Standard import of function from kaggle



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
import sys

sys.version
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import glob

import json

import re

import matplotlib.pyplot as plt

import matplotlib.cm as cm



from sklearn.cluster import MiniBatchKMeans

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from nltk.stem.porter import PorterStemmer

import nltk
# First read what is in the Folder

!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

#root_path = '/kaggle/input/cord19researchchallenge20200320/2020-03-13'

#metadata_path = f'{root_path}/all_sources_metadata_2020-03-13.csv'





meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
# Get all JSON paths

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
len(meta_df.publish_time)
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

first_row = FileReader(all_json[0])

print(first_row)
def get_breaks(content, length):

    data = ""

    words = content.split(' ')

    total_chars = 0



    # add break every length characters

    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': [],'doi':[]}#,'doi':[]}#,'publish_time':[],'doi':[]}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 300 words append with ...

        info = content.abstract.split(' ')[:100]

        summary = get_breaks(' '.join(info), 40)

        dict_['abstract_summary'].append(summary + "...")

    else:

        # abstract is short enough

        summary = get_breaks(content.abstract, 40)

        dict_['abstract_summary'].append(summary)

        

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    

    try:

        # if more than one author

        authors = meta_data['authors'].values[0].split(';')

        if len(authors) > 2:

            # more than 2 authors, may be problem when plotting, so take first 2 append with ...

            dict_['authors'].append(". ".join(authors[:2]) + "...")

        else:

            # authors will fit in plot

            dict_['authors'].append(". ".join(authors))

    except Exception as e:

        # if only one author - or Null valie

        dict_['authors'].append(meta_data['authors'].values[0])

    

    # add the title information, add breaks when needed

    try:

        title = get_breaks(meta_data['title'].values[0], 40)

        dict_['title'].append(title)

    # if title was not provided

    except Exception as e:

        dict_['title'].append(meta_data['title'].values[0])

        

        # add the DOI information, add breaks when needed

    try:

        doi = get_breaks(meta_data['doi'].values[0], 40)

        dict_['doi'].append(doi)

    # if doi was not provided

    except Exception as e:

        dict_['doi'].append(meta_data['doi'].values[0])



    # add the journal information

    dict_['journal'].append(meta_data['journal'].values[0])

    #dict_['doi'].append(meta_data['doi'].values[0])

    #dict_['publish_time'].append(meta_data['publish_time'].values[0])

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary','doi'])#,'doi'])#,'publish_time','doi'])

df_covid.head()
df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))

df_covid.head()
# Get info about all the columns

df_covid.info()
df_covid['abstract'].describe(include='all')

df_covid.head()
df_covid.describe()
df_covid.dropna(inplace=True)

df_covid.info()
def lower_case(input_str):

    input_str = input_str.lower()

    return input_str



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))
COVID_19 = df_covid.body_text.map(lambda text: "covid-19" in text)

covid_192 = df_covid.body_text.map(lambda text: "covid19" in text)

tot = covid_192|COVID_19

tot.sum()
df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

df_covid["body_text_stem"] = df_covid["body_text"].map(lambda txt: ' '.join(stemmer.stem(token) for token in nltk.word_tokenize(txt)))

df_covid["abstract"] = df_covid["abstract"].map(lambda txt: ' '.join(stemmer.stem(token) for token in nltk.word_tokenize(txt)))

df_covid.head(10)
ID = df_covid.body_text.map(lambda text: "intellectual disab" in text)

LR = df_covid.body_text.map(lambda text: "learning disab" in text)

MR = df_covid.body_text.map(lambda text: "mental retardation" in text)

CD = df_covid.body_text.map(lambda text: "cognitive disab" in text)

MD = df_covid.body_text.map(lambda text: "mental disab" in text)

DS = df_covid.body_text.map(lambda text: "down syndrome" in text)

FX = df_covid.body_text.map(lambda text: "fragile x " in text)

PWS = df_covid.body_text.map(lambda text: "prader willi " in text)

WS = df_covid.body_text.map(lambda text: "williams syndrome" in text)

FASD = df_covid.body_text.map(lambda text: "fetal alcohol spectrum disorder" in text)

RETT = df_covid.body_text.map(lambda text: "rett syndrome" in text)

VCFC = df_covid.body_text.map(lambda text: "velo-cardio-facial syndrome" in text)

ANG = df_covid.body_text.map(lambda text: "angelman syndrome" in text)

TSC = df_covid.body_text.map(lambda text: "tuberous sclerosis complex" in text)

CDLS = df_covid.body_text.map(lambda text: "cornelia de lange syndrome" in text)



ID_tot = ID.sum()

LR_tot = LR.sum()

MR_tot = MR.sum()

CD_tot = CD.sum()

MD_tot = MD.sum()

DS_tot = DS.sum()

FX_tot = FX.sum()

PWS_tot = PWS.sum()

WS_tot = WS.sum()

FASD_tot = FASD.sum()

RETT_tot = RETT.sum()

VCFC_tot = VCFC.sum()

ANG_tot = ANG.sum()

TSC_tot = TSC.sum()

CDLS_tot = CDLS.sum()



ID_appearances = pd.Series([ID_tot,LR_tot,MR_tot,CD_tot, MD_tot, DS_tot,FX_tot,PWS_tot, WS_tot,FASD_tot, RETT_tot,VCFC_tot, ANG_tot, TSC_tot, CDLS_tot],

                           index = ["intellectual disability","learning disability","mental retardation" ,"cognitive disability","mental disability","down syndrome" , "fragile x ",

                                    "prader willi ","williams syndrome","fetal alcohol spectrum disorder","rett syndrome","velo-cardio-facial syndrome",

                                   "angelman syndrome","tuberous sclerosis complex","cornelia de lange syndrome"],name = "Texts with certain word in full text")

ID_appearances
# Get the indexes of all papers with one of the above and make a new dataframe with the ID papers

papers_ID = ID|LR|MR|CD|MD|DS|FX|PWS|WS|RETT|VCFC|ANG|TSC|CDLS # Is | the correct operator?

df_covid_ID = df_covid[papers_ID]

df_covid_no_ID = df_covid[-papers_ID]

print(len(df_covid_ID))

print(len(df_covid))

print(len(df_covid_no_ID))

df_covid_ID.head()



kind = list()

for i in papers_ID:

    if i == True:

        kind.append("ID_paper")

    else:

        kind.append("Non_ID_paper")

        

df_covid["paper_kind"] = kind

df_covid.head(10)

df_covid.to_csv("df_covid_stemmed.csv",header = True, index = True)

#df_covid_ID.to_csv("df_covid_ID_stemmed.csv",header = True, index = True)

#df_covid_no_ID.to_csv("df_covid_no_ID_stemmed.csv",header = True, index = True)
#root_path = '/kaggle/input/stemmed'

#data_path = f'{root_path}/df_covid_stemmed.csv'

#data_path_ID = f'{root_path}/df_covid_ID_stemmed.csv'

#data_path_no_ID = f'{root_path}/df_covid_ID_stemmed.csv'

#df_covid = pd.read_csv(data_path)

#df_covid_ID = pd.read_csv(data_path_ID)

#df_covid_no_ID = pd.read_csv(data_path_no_ID)
def get_stop_words(stop_file_path):

    """load stop words """

    

    with open(stop_file_path, 'r', encoding="utf-8") as f:

        stopwords = f.readlines()

        stop_set = set(m.strip() for m in stopwords)

        return frozenset(stop_set)





stopwords=get_stop_words('/kaggle/input/stopwords/stopwords_joep.txt')
tfidf = TfidfVectorizer(

    min_df = 5,

    max_df = 0.95, #try at 0.95

    max_features = 8000,

    stop_words = stopwords)

tfidf.fit(df_covid.body_text_stem)

text = tfidf.transform(df_covid_ID.body_text_stem)
def find_optimal_clusters(data, max_k):

    iters = range(2, max_k+1, 1)

    

    sse = []

    for k in iters:

        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)

        print('Fit {} clusters'.format(k))

        

    f, ax = plt.subplots(1, 1)

    ax.plot(iters, sse, marker='o')

    ax.set_xlabel('Cluster Centers')

    ax.set_xticks(iters)

    ax.set_xticklabels(iters)

    ax.set_ylabel('SSE')

    ax.set_title('SSE by Cluster Center Plot')

    

find_optimal_clusters(text, 10)
clusters = MiniBatchKMeans(n_clusters=5, init_size=1024, batch_size=2048, random_state=20).fit_predict(text)
def plot_tsne_pca(data, labels):

    max_label = max(labels+1)

    max_items = np.random.choice(range(data.shape[0]), size = data.shape[0], replace=False)

    

    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())

    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))

    

    

    idx = np.random.choice(range(pca.shape[0]), size = data.shape[0], replace=False)

    label_subset = labels[max_items]

    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]

    

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)

    ax[0].set_title('PCA Cluster Plot')

    

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)

    ax[1].set_title('TSNE Cluster Plot')

    

plot_tsne_pca(text, clusters)
def get_top_keywords(data, clusters, labels, n_terms):

    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    

    for i,r in df.iterrows():

        print('\nCluster {}'.format(i))

        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

            

get_top_keywords(text, clusters+1, tfidf.get_feature_names(), 10)
df_covid_ID["Cluster"] = clusters
df_covid_ID.head()
df_ID_articles = df_covid_ID[["authors","title","journal" ,"Cluster"]]

pd.options.display.max_rows = 259

pd.options.display.max_colwidth = 125

df_ID_articles