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

from matplotlib.lines import Line2D

from matplotlib.legend_handler import HandlerPatch

from PIL import Image

import io
root_path = '/kaggle/input/stemmed'

data_path = f'{root_path}/df_covid_stemmed.csv'

df_covid = pd.read_csv(data_path)



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

papers_ID = ID|LR|MR|CD|MD|DS|FX|PWS|WS|RETT|VCFC|ANG|TSC|CDLS 

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

df_covid.head(10) # Check that paper with index 8 is an ID paper

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
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

stemmer.stem("quarantaine")
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

    # Save the image in memory in PNG format

    png1 = io.BytesIO()

    f.savefig(png1, format="png")



    # Load this image into PIL

    png2 = Image.open(png1)



    # Save as TIFF

    png2.save("Figure_1.tiff")

    png1.close()

    

find_optimal_clusters(text, 10)
clusters = MiniBatchKMeans(n_clusters=5, init_size=1024, batch_size=2048, random_state=20).fit_predict(text)
def plot_tsne_pca(data, labels):

    max_label = max(labels+1)

    max_items = np.random.choice(range(data.shape[0]), size = data.shape[0], replace=False)

    

    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())

    #tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))

    

    

    idx = np.random.choice(range(pca.shape[0]), size = data.shape[0], replace=False)

    label_subset = labels[max_items]

    label_subset2 = [cm.hsv(i/max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 1)

    ax.scatter(pca[idx, 0], pca[idx, 1], c=label_subset2, label=label_subset)

    ax.set_title('PCA Cluster Plot')

    ax.set_xlabel("PCA 1")

    ax.set_ylabel("PCA 2") 

    plt.show()



    

plot_tsne_pca(text, clusters)
def plot_pca(data,labels):

    max_label = max(labels+1)  

    pca = PCA(n_components=2).fit_transform(data.todense())

    idx = range(pca.shape[0])

    label_subset = labels

    label_color = [cm.hsv(i/max_label) for i in label_subset[idx]]

    fig, ax = plt.subplots()

    for i in range(0,len(labels)):

        scatter = ax.scatter(pca[i,0],pca[i,1],c= label_color[i],label = labels[i]+1)

    #scatter = ax.scatter(pca[idx, 0], pca[idx, 1], c=label_color, label = labels)

    ax.set_title('PCA Cluster Plot')

    ax.set_xlabel("PCA 1")

    ax.set_ylabel("PCA 2") 

    ax.grid(True)

    legend_elements = [Line2D([], [], marker='o', color=(1.0, 0.0, 0.0, 1.0), label='1',

                          markersize=5, linestyle='None'),

                      Line2D([], [], marker='o', color=(0.8187488187488188, 1.0, 0.0, 1.0), label='2',

                          markersize=5, linestyle='None'),

                      Line2D([], [], marker='o', color=(0.0, 1.0, 0.3625004724974957, 1.0), label='3',

                          markersize=5, linestyle='None'),

                      Line2D([], [], marker='o', color=(0.0, 0.4562523625023627, 1.0, 1.0), label='4',

                          markersize=5, linestyle='None'),

                      Line2D([], [], marker='o', color=(0.724998818748819, 0.0, 1.0, 1.0), label='5',

                         markersize=5, linestyle='None')

]

    ax.legend(handles=legend_elements, loc='lower right',title="Cluster")

    plt.show()

    # Save the image in memory in PNG format

    png1 = io.BytesIO()

    fig.savefig(png1, format="png")



    # Load this image into PIL

    png2 = Image.open(png1)



    # Save as TIFF

    png2.save("Figure_2.tiff")

    png1.close()



plot_pca(text, clusters)
def get_top_keywords(data, clusters, labels, n_terms):

    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    

    for i,r in df.iterrows():

        print('\nCluster {}'.format(i))

        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

        print(sum(clusters == i))

            

get_top_keywords(text, clusters+1, tfidf.get_feature_names(), 10)

df_covid_ID["Cluster"] = clusters+1 #Needed for indexng python
df_covid_ID.head()
# get metadata

root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()



df_covid_ID["doi"] = 0

df_covid_ID["year"] = 0

# ugly way with loops, but works

for i in range(0, len(df_covid_ID["paper_id"])):

    for j in range(0,len(meta_df["sha"])):

        if df_covid_ID.paper_id.iloc[i] == meta_df.sha.iloc[j]:

            df_covid_ID.doi.iloc[i] = meta_df.doi.iloc[j]

            df_covid_ID.year.iloc[i] = meta_df.publish_time.iloc[j]

    
df_ID_articles = df_covid_ID[["authors","title","journal" ,"Cluster","doi","year"]]

pd.options.display.max_rows = 259

pd.options.display.max_colwidth = 125

df_ID_articles