import covid19_tools as cv19

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

import numpy as np

import gensim

from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import cdist

from IPython.core.display import display, HTML



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

pd.options.mode.chained_assignment = None 
METADATA_FILE = '/kaggle/input/CORD-19-research-challenge/metadata.csv'



# load metadata

meta = cv19.load_metadata(METADATA_FILE)

# add covid19 tags

meta, covid19_counts = cv19.add_tag_covid19(meta)
# select only Covid-19 related articles

meta = meta[meta.tag_disease_covid19]
SOC_ETHIC_TERMS = ['social',

'social concern',

'ethical principles',

'ethical concern',

'ethical framework',

'restriction',

'movement',

'restriction of movement',

'movement restriction',

'tracking',

'tracing',

'freedom',

'assembly',

'gathering',

'detention',

'vaccination',

'censorship',

'internet',

'access',

'medical treatment',

'disinformation',

'misinformation',

'fake news',

'isolation',

'individual rights',

'human rights',

'liberty',

'self-determination',

'ethical',

'antiviral',

'force',

'force measures',

'privacy',

'surveillance',

'digital rights',

'democracy',

'discrimination',

'anti-asian',

'unemployment',

'politics',

'compliance',

'bed shortage',

'ICU bed shortage',

'hospitals overloaded',

'public health']
# add social and ethic tags

meta, soc_ethic_counts = cv19.count_and_tag(meta,SOC_ETHIC_TERMS,'soc_ethic')

# include only social and ethic related articles

meta_rel = meta[meta.tag_soc_ethic]
meta_rel.info()
meta_rel.head(5)
print('Loading full text for articles')

full_text_repr = cv19.load_full_text(meta_rel,'../input/CORD-19-research-challenge')
full_text_repr[0]
def get_body_text(full_text_repr):

    body_text = []

    for article in full_text_repr:

        text = [body_text['text'] for body_text in article['body_text']]

        body_text.append(''.join(text))

    return body_text



# extract body text

body_text_repr = get_body_text(full_text_repr)



# store body text if exists

full_text_ids = [article['paper_id'] for article in full_text_repr]

meta_rel['full_text'] = None

meta_rel['full_text'] = meta_rel['sha'].apply(lambda x: full_text_ids.index(x) if x in full_text_ids else -1)

meta_rel['full_text'] = meta_rel['full_text'].apply(lambda x: body_text_repr[x] if x != -1 else None)
meta_rel.head(5)
print('Number of articles without title:')

print(meta_rel['title'].isnull().sum())

print('Number of articles without abstract:')

print(meta_rel['abstract'].isnull().sum())

print('Number of articles without full text:')

print(meta_rel['full_text'].isnull().sum())
def read_corpus(df, column, tokens_only=False):

    for i, line in enumerate(df[column]):

        # get text tokens

        tokens = gensim.utils.simple_preprocess(line)

        if tokens_only:

            yield tokens

        else:

            # for training data, add tags

            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
# we use as training data the full text of the articles

train_corpus = meta_rel[meta_rel['full_text'].notnull()]

train_corpus = list(read_corpus(train_corpus, 'full_text'))
train_corpus[0]
# train the model

model = gensim.models.doc2vec.Doc2Vec(vector_size=512, min_count=2, epochs=20, seed=42, workers=3)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
# this functions calculates text embedding using our trained model

def get_doc_vector(doc):

    tokens = gensim.utils.simple_preprocess(doc)

    embedding = model.infer_vector(tokens)

    return embedding
# get the full text embeddings, if full text is missing then set embedding to 512 dimensional vector of Nones

doc2vec_full_text_embedding = meta_rel.apply(lambda x: get_doc_vector(x['full_text']) if 

                                             pd.notnull(x['full_text']) else np.full(512, None), axis = 1)



meta_rel['doc2vec_full_text_embedding'] = doc2vec_full_text_embedding

del doc2vec_full_text_embedding
# download the Universal Sentence Encoder module and uncompress it to the destination folder. 

!mkdir /kaggle/working/universal_sentence_encoder/

!curl -L 'https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed' | tar -zxvC /kaggle/working/universal_sentence_encoder/
# load the Universal Sentence Encoder module

use_embed = hub.load('/kaggle/working/universal_sentence_encoder/')



# this function calculates text embedding using Universal Sentence Encoder

def get_use_embedding(text):

    embedding = use_embed([text])[0]

    return embedding.numpy()
# get the title embeddings, if title is missing then set embedding to 512 dimensional vector of Nones

use_title_embedding = meta_rel.apply(lambda x: get_use_embedding(x['title']) if pd.notnull(x['title']) 

                                else np.full(512, None), axis = 1)



meta_rel['use_title_embedding'] = use_title_embedding

del use_title_embedding
# get the abstract embeddings, if abstract is missing then set embedding to 512 dimensional vector of Nones

use_abstract_embedding = meta_rel.apply(lambda x: get_use_embedding(x['abstract']) if pd.notnull(x['abstract']) 

                                else np.full(512, None), axis = 1)

meta_rel['use_abstract_embedding'] = use_abstract_embedding

del use_abstract_embedding
# compute cosine similarity metric between embeddings, note that if embedding is None similarity is nan

def cosine_similarity(embedding, embeddings):

    emb_shape = embedding.shape[0]

    embeddings = np.stack(embeddings.to_numpy(), axis = 0)

    similarities = 1 - cdist(embedding.reshape(1,emb_shape), embeddings, 'cosine')

    return similarities[0].astype('float32')
# compute the similarity metrics between each embedding and the rest

use_title_simil = meta_rel['use_title_embedding'].apply(lambda x: cosine_similarity(x , meta_rel['use_title_embedding']))

meta_rel['use_title_cosine_sim'] = use_title_simil

del use_title_simil



use_abs_simil = meta_rel['use_abstract_embedding'].apply(lambda x: cosine_similarity(x , meta_rel['use_abstract_embedding']))

meta_rel['use_abstract_cosine_sim'] = use_abs_simil

del use_abs_simil



doc2vec_full_text_simil = meta_rel['doc2vec_full_text_embedding'].apply(lambda x: cosine_similarity(x , meta_rel['doc2vec_full_text_embedding']))

meta_rel['doc2vec_full_text_cosine_sim'] = doc2vec_full_text_simil

del doc2vec_full_text_simil
meta_rel.head()
# save DataFrame with embeddings and similarities

!mkdir /kaggle/working/output_data/

meta_rel.to_pickle('/kaggle/working/output_data/data.pkl')
# compute similarity matrix between the embeddings, note that if embedding is None similarity is nan

def similarity_matrix(embeddings):

    embeddings = np.stack(embeddings.to_numpy(), axis = 0)

    similarities = 1 - cdist(embeddings, embeddings, 'cosine')

    return similarities.astype('float32')
# get similarity matrices for titles, abstract and full texts

use_title_sim_matrix = similarity_matrix(meta_rel['use_title_embedding'])

use_abstract_sim_matrix = similarity_matrix(meta_rel['use_abstract_embedding'])

doc2vec_full_text_sim_matrix = similarity_matrix(meta_rel['doc2vec_full_text_embedding'])
use_title_sim_matrix
# create dataframes of similarities

index_id = meta_rel['sha'].values

use_title_sim_matrix_df = pd.DataFrame(use_title_sim_matrix, index = index_id, columns = index_id)

use_abstract_sim_matrix_df = pd.DataFrame(use_abstract_sim_matrix, index = index_id, columns = index_id)

doc2vec_full_text_sim_matrix_df = pd.DataFrame(doc2vec_full_text_sim_matrix, index = index_id, columns = index_id)
use_title_sim_matrix_df.head(5)
# save similarity matrices as Numpy arrays and Pandas DataFrames

output_dir = '/kaggle/working/output_data/'



np.save(output_dir + 'title_sim.npy',use_title_sim_matrix)

np.save(output_dir + 'abstract_sim.npy',use_abstract_sim_matrix)

np.save(output_dir + 'full_text_sim.npy',doc2vec_full_text_sim_matrix)



use_title_sim_matrix_df.to_pickle(output_dir + 'title_sim_df.pkl')

use_abstract_sim_matrix_df.to_pickle(output_dir + 'abstract_sim_df.pkl')

doc2vec_full_text_sim_matrix_df.to_pickle(output_dir + 'full_text_sim_df.pkl')
task_1 = ('Efforts to articulate and translate existing ethical principles and standards to salient '

            'issues in COVID-2019')

task_2 = ('Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise '

            'and coordinate to minimize duplication of oversight')

task_3 = 'Efforts to support sustained education, access, and capacity building in the area of ethics '

task_4 = ('Efforts to establish a team at WHO that will be integrated within multidisciplinary research '

            'and operational platforms and that will connect with existing and expanded global networks '

            'of social sciences.')

task_5 = ('Efforts to develop qualitative assessment frameworks to systematically collect information '

            'related to local barriers and enablers for the uptake and adherence to public health measures ' 

            'for prevention and control. This includes the rapid identification of the secondary impacts of '

            'these measures. (e.g. use of surgical masks, modification of health seeking behaviors for '

            'SRH, school closures)')

task_6 = ('Efforts to identify how the burden of responding to the outbreak and implementing public '

            'health measures affects the physical and psychological health of those providing care for '

            'Covid-19 patients and identify the immediate needs that must be addressed.')

task_7 = ('Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation '

            'and rumor, particularly through social media.')
tasks = [task_1, task_2, task_3, task_4, task_5, task_6, task_7]

# compute USE embeddings for tasks

use_tasks_embeddings = [get_use_embedding(task) for task in tasks]
def get_train_embeddings(df, column):

    # remove rows in which embedding is vector of Nones (i.e. title or abstract is missing)

    train_embeddings = df[column].apply(lambda x: None if x[0] is None else x).dropna()

    index = train_embeddings.index

    train_embeddings = train_embeddings.to_list()

    # return list of not None embeddings and DataFrame without articles whose embedding is None

    return train_embeddings, df.loc[index]



# auxiliar function to print info about found neighbors

def print_neighbors_info(tasks, meta_rel_mod, neigh_dist, neigh_indices):

    for i, task in enumerate(tasks):

        print("-"*80, f'\n\nTask = {task}\n')

        df =  meta_rel_mod.iloc[neigh_indices[i]]

        abstracts = df['abstract']

        titles = df['title']

        dist = neigh_dist[i]

        for neighbour in range(len(dist)):

            print(f'Distance = {neigh_dist[i][neighbour]:.4f} \n')

            print(f'Title: {titles.iloc[neighbour]} \n\nAbstract: {abstracts.iloc[neighbour]}\n\n')
train_data, meta_rel_mod = get_train_embeddings(meta_rel, 'use_title_embedding')

nn_model = NearestNeighbors().fit(train_data)

neigh_dist, neigh_indices = nn_model.kneighbors(use_tasks_embeddings, n_neighbors=3)
print_neighbors_info(tasks, meta_rel_mod, neigh_dist, neigh_indices)
train_data, meta_rel_mod = get_train_embeddings(meta_rel, 'use_abstract_embedding')

# options: 'use_title_embedding', 'use_abstract_embedding', 'doc2vec_full_text_embedding'

nn_model = NearestNeighbors().fit(train_data)

neigh_dist, neigh_indices = nn_model.kneighbors(use_tasks_embeddings, n_neighbors=3)
print_neighbors_info(tasks, meta_rel_mod, neigh_dist, neigh_indices)