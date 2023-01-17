import json

import requests

import time

import os

import pickle

import csv

import numpy as np
def read_prepared_specter_embeddings(csv_path):

    with open(csv_path, newline='') as csvfile:

        l = list(csv.reader(csvfile, delimiter=','))

    # The entries have as their first value a unique id (hash),

    # followed by the actual vector. Extract this into a dict

    # for better access.

    embeddingsDict = {}

    uidList = []

    for paperE in l:

        cp = list(paperE)

        uid = cp.pop(0)

        uidList.append(uid)

        newList = np.array([float(emb) for emb in cp])

        embeddingsDict[uid] = newList

    return uidList, embeddingsDict
uidList, cord19_dict = read_prepared_specter_embeddings(

    '/kaggle/input/CORD-19-research-challenge/cord19_specter_embeddings_2020-04-10/cord19_specter_embeddings_2020-04-10.csv')
myGuardianAPIkey = None

GuardianArticlesStoragePath = '/kaggle/input/guardian-covid19/Guardian_Covid-19' # TODO: absolute kaggle path
def grabArticles(nArticles, sectionRestr = None):

    for page in range(nArticles//10):

        searchTerm = 'Covid-19'

        requestParams = { 'q': searchTerm 

                        , 'show-fields': 'bodyText'

                        , 'page': page+1

                        , 'api-key': myGuardianAPIkey

                        }

        if sectionRestr is not None:

            requestParams['section'] = sectionRestr

        relevantArticlesResp = requests.get(

                    'https://content.guardianapis.com/search'

                  , requestParams ).json()

            

        for entry in relevantArticlesResp['response']['results']:

            fname = '_'.join(entry['webUrl'].split('/')[3:]) + '.json'

            textContent = entry['fields']['bodyText']

            # Pack the content into a dictionary structure compatible

            # for the SemanticScholar encoder. We use the body text as

            # both the article text and the abstract: arguably, a news

            # article is already more like the abstract of a scientific

            # article, since it is written in a higher-level style

            # suitable for non-experts.

            jsonFormatted = {

                     'url': entry['webUrl']

                   , 'title': entry['webTitle']

                   , 'abstract': textContent

                   , 'body_text': textContent

                   }

            with open(GuardianArticlesStoragePath+'/'+fname, 'w') as fh:

                json.dump(jsonFormatted, fh)

        time.sleep(10) # avoid too fast requests to API
# grabArticles(150)            # Uncomment to retrive articles anew

# grabArticles(50, 'science')
URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"

MAX_BATCH_SIZE = 16
# Source: https://github.com/allenai/paper-embedding-public-apis

def chunks(lst, chunk_size=MAX_BATCH_SIZE):

    """Splits a longer list to respect batch size"""

    for i in range(0, len(lst), chunk_size):

        yield lst[i : i + chunk_size]

        

def embed(papers):

    embeddings_by_paper_id: Dict[str, List[float]] = {}



    for chunk in chunks(papers):

        # Allow Python requests to convert the data above to JSON

        response = requests.post(specterURL, json=chunk)



        if response.status_code != 200:

            raise RuntimeError("Sorry, something went wrong, please try later!")



        for paper in response.json()["preds"]:

            embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]



    return embeddings_by_paper_id



def loadJsonPaper(fp):

    paperId = os.path.basename(fp).split('.')[0]

    with open(fp, 'r') as fh:

        content = json.load(fh)

    content['paper_id'] = paperId

    return content
relevantNewsArticleIds = ['world_2020_mar_16_health-experts-criticise-nhs-advice-to-take-ibuprofen-for-covid-19',

                          'science_2020_mar_25_can-chloroquine-really-help-treat-coronavirus-patients',

                          'science_2020_feb_20_doctors-hiv-ebola-drugs-coronavirus-cure-covid-19',

                          'sport_2020_mar_26_ecb-steve-elworthy-cricket-coronavirus',

                          'world_2020_mar_30_fall-in-covid-19-tests-putting-lives-at-risk-critics-claim',

                          'world_2020_mar_29_ventilator-challenge-uk-to-start-production-in-covid-19-fight']
# We have already saved the embeddings previously. Set this to False

# to request them anew.

loadPrecomputedNewsEmbeddings = True



def getRelevantNewsEmbeddings():

    embeddingDir = GuardianArticlesStoragePath+'_specter'

    if not loadPrecomputedNewsEmbeddings:

        os.makedirs(embeddingDir, exist_ok=True)



    if loadPrecomputedNewsEmbeddings:

        articleDict = {}

        for articleId in relevantNewsArticleIds:

            articleDict[articleId] = pickle.load(

              open(embeddingDir+'/'+articleId, 'rb'))

        return articleDict

    else:

        relevantNewsContent = [

         loadJsonPaper("Guardian_Covid-19/"+articleId+".json")

            for articleId in relevantNewsArticleIds ]

        embedded = embed(relevantNewsContent)

        for articleId, embd in embedded.items():

            fp = embeddingDir+'/'+articleId

            pickle.dump(embd, open(fp, 'wb'))

        return embedded



def getRelevantNewsEmbedIDarray():

    embeddingDir = GuardianArticlesStoragePath+'_specter'

    if not loadPrecomputedNewsEmbeddings:

        os.makedirs(embeddingDir, exist_ok=True)

    if loadPrecomputedNewsEmbeddings:

        articleArray = np.zeros((len(relevantNewsArticleIds),768))

        IdList = []

        for i, articleId in enumerate(relevantNewsArticleIds):

            articleArray[i] = pickle.load(

              open(embeddingDir+'/'+articleId, 'rb'))

            IdList.append(articleId)

        return IdList, articleArray
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
def pcadimred(data):

    pca = PCA(n_components=0.8)

    return pca.fit_transform(data)
cord19_full = np.zeros((len(uidList),len(cord19_dict[uidList[0]])))

for i,uid in enumerate(uidList):

    cord19_full[i,:] = cord19_dict[uid]
newsidList, news_embeddings = getRelevantNewsEmbedIDarray()
cord19_news_full = np.vstack((cord19_full,news_embeddings))

print('Original dimensions: ', np.shape(cord19_news_full))

cord19_news_red = pcadimred(cord19_news_full)

print('Reduced dimensions using PCA: ', np.shape(cord19_news_red))
def tsnedimred(data):

    tsne = TSNE(verbose=1)

    return tsne.fit_transform(data)
# Uncomment to calculate again (takes a few minutes)

#cord19_news_2d = tsnedimred(cord19_news_red)
# Saving and loading cord19_news_2d to disk

cord19_news_2d = np.load('/kaggle/input/covidtemp/cord19_news_2d.npy')

#cord19_news_2d = np.load('/kaggle/working/cord19_news_2d.npy')

np.save('/kaggle/working/cord19_news_2d', cord19_news_2d)
cord19_2d = (cord19_news_2d[:-len(newsidList),0], cord19_news_2d[:-len(newsidList),1])

news_2d = (cord19_news_2d[len(uidList):,0], cord19_news_2d[len(uidList):,1])

data = (cord19_2d, news_2d)

colors = ("blue", "red")

groups = ("CORD-19", "News articles")

size = (1,30)



# Create plot

fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(1, 1, 1, facecolor="1.0")



for data, color, group, size in zip(data, colors, groups, size):

    x, y = data

    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=size, label=group)



plt.title('t-SNE of CORD-19 and 6 news articles')

plt.legend(loc=2)

plt.savefig("/kaggle/working/t-sne_cord19_news.png")

plt.show()
cord19_2d = (cord19_news_2d[:-len(newsidList),0], cord19_news_2d[:-len(newsidList),1])

news_2d = np.asarray((cord19_news_2d[len(uidList):,0], cord19_news_2d[len(uidList):,1]))



data = (cord19_2d, news_2d[:,0], news_2d[:,1], news_2d[:,2], news_2d[:,3], news_2d[:,4], news_2d[:,5])

colors = ("blue", "red", "green", "black", "yellow", "cyan", "orange")

groups = ("CORD-19", newsidList[0], newsidList[1], newsidList[2], newsidList[3], newsidList[4], newsidList[5])

size = (1,30,30,30,30,30,30)



# Create plot

fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(1, 1, 1, facecolor="1.0")



for data, color, group, size in zip(data, colors, groups, size):

    x, y = data

    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=size, label=group)



plt.title('t-SNE of CORD-19 and 6 news articles')

plt.legend(loc='lower right')

plt.savefig("/kaggle/working/t-sne_cord19_news_withlabels.png")

plt.show()
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

import pprint
# Hack to force abstract column to be wider

abstract_long_name = '_________________abstract_of_research_paper_________________'



def read_metadata(metadata_path):

    meta_df = pd.read_csv(metadata_path, dtype={

        'pubmed_id': str,

        'Microsoft Academic Paper ID': str, 

        'doi': str

    })

    # Hack to force abstract column to be wider

    meta_df.rename(

        columns = {'abstract': abstract_long_name}, 

        inplace = True)

    return meta_df

metadata = read_metadata('/kaggle/input/CORD-19-research-challenge/metadata.csv')
def get_cosine_similarity(feature_vec_1, feature_vec_2):

    """https://medium.com/@Intellica.AI/comparison-of-different-word-embeddings-on-text-similarity-a-use-case-in-nlp-e83e08469c1c"""

    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]



def get_cosine_similarities(cord_papers_embeddings, news_embedding):

    d = {}

    for uid, paper_embedding in cord_papers_embeddings.items():

        d[uid] = get_cosine_similarity(news_embedding, paper_embedding)

    return d



def add_cosine_sim_to_metadata_df(metadata_df, similarities_dict):

    metadata_df_c = metadata_df.copy() # Deep copy

    metadata_df_c['cosine_similarity'] = metadata_df_c['cord_uid'].map(similarities_dict)

    return metadata_df_c.sort_values(by='cosine_similarity', ascending=False)
import textwrap



def make_clickable(val):

    # target _blank to open new window

    return '<a target="_blank" href="{}">URL</a>'.format(val)



def change_abstract_name(val):

    cell_length = 800

    val = str(val)

    if len(val) > cell_length:

        val = textwrap.wrap(val, cell_length)[0] + '...'

    return val



def url_style(metadata_df):

    return metadata_df.style.format({'url': make_clickable, 

                                     abstract_long_name: change_abstract_name})



cols_to_display = ['cosine_similarity', 'publish_time', 'url', 'title', 

                   abstract_long_name, 'authors', 'journal', 'cord_uid']



def show_k_most_similar_papers(metadata_df, k):

    return url_style(metadata_df.head(k).loc[:, cols_to_display])



def show_k_least_similar_papers(metadata_df, k):

    return url_style(metadata_df.tail(k).loc[:, cols_to_display])
news_embeddings = getRelevantNewsEmbeddings()

pp = pprint.PrettyPrinter()

pp.pprint(list(news_embeddings.keys()))
ibuprofen_embedding = np.array(news_embeddings['world_2020_mar_16_health-experts-criticise-nhs-advice-to-take-ibuprofen-for-covid-19'])

ibuprofen_similarity = add_cosine_sim_to_metadata_df(metadata, get_cosine_similarities(cord19_dict, ibuprofen_embedding))
show_k_most_similar_papers(ibuprofen_similarity, 10)
show_k_least_similar_papers(ibuprofen_similarity, 5)
cricket_embedding = np.array(news_embeddings['sport_2020_mar_26_ecb-steve-elworthy-cricket-coronavirus'])

cricket_similarity = add_cosine_sim_to_metadata_df(metadata, get_cosine_similarities(cord19_dict, cricket_embedding))
show_k_most_similar_papers(cricket_similarity, 10)
ebola_embedding = np.array(news_embeddings['science_2020_feb_20_doctors-hiv-ebola-drugs-coronavirus-cure-covid-19'])

ebola_similarity = add_cosine_sim_to_metadata_df(metadata, get_cosine_similarities(cord19_dict, ebola_embedding))
show_k_most_similar_papers(ebola_similarity, 10)