# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# There are too many paths and printing them takes
# up too much space so I don't do this normally
if 1==0: 
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            pass
            #print(os.path.join(dirname, filename))
# Progress bar
import tqdm

# Word2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec,TaggedDocument

from nltk.tokenize import word_tokenize 
from scipy.spatial.distance import cdist
biorxiv_clean = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")
clean_comm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")
clean_noncomm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")
clean_pmc = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv")

all_data = pd.concat([biorxiv_clean, clean_comm_use, clean_noncomm_use, clean_pmc]).reset_index(drop=True)

all_data.head()
del biorxiv_clean,clean_comm_use,clean_noncomm_use,clean_pmc
import gc
gc.collect()
print("Number of Rows in Table: %i" % len(all_data))
print("Number of Titles: %i " % all_data['title'].count())
print("Number of Abstracts: %i " % all_data['abstract'].count())
print("Number of Texts: %i " % all_data['text'].count())
# replace empty text with empty strings
all_title = all_data.title.str.replace('\n\n', ' ')
non_na = all_title.notna()
non_na_title = all_title[non_na]
non_na_paper_ids = all_data[non_na]['paper_id']
non_na_title_list = list(
    map(
        lambda x: word_tokenize(x), non_na_title.values
    ))
documents = [TaggedDocument(doc, [non_na_paper_ids.values[i]]) for i, doc in enumerate(non_na_title_list)]
model = Doc2Vec(documents,vector_size = 300,window=2, min_count=1, workers=4)
document_dict = {}
for idx, text_df in tqdm.tqdm(all_data[non_na][["paper_id", "title"]].iterrows()):
    text = text_df['title'].replace("\n\n", ' ')
    document_dict[text_df['paper_id']] = model.docvecs[text_df['paper_id']]
print(len(document_dict))
print(list(document_dict.values())[0].shape)
document_embeddings_df = pd.DataFrame.from_dict(document_dict, orient="index")
document_embeddings_df.head()
def cos_sim(text):

    mean_word_vec = pd.np.stack(word_vector_list, axis=1).mean(axis=1)
    
    # compute similarity
    doc_sim = (
        1-cdist(
            document_embeddings_df.values,
            mean_word_vec,
            'cosine'
        )
    )
    # convert result to a date frame
    document_sim_df = (
        pd.DataFrame(doc_sim, columns=["cos_sim"])
        .assign(document_id=list(document_embeddings_df.index))
    )
    # sort from most similar to least
    document_sim_df = document_sim_df.sort_values("cos_sim", ascending=False)
    
    # perform left-join to get information about the documents
    doc_sim_meta_df = document_sim_df.merge(all_data,
                      how='left',
                     left_on='document_id',
                     right_on='paper_id')
    return(doc_sim_meta_df)
doc_sim = (
    1 - cdist(
        document_embeddings_df.values,
        [model.wv['airborne']],
        'cosine'
    )
)
doc_sim.shape
document_sim_df = (
    pd.DataFrame(doc_sim, columns=["cos_sim"])
    .assign(document_id=list(document_embeddings_df.index))
)
document_sim_df.sort_values(by='cos_sim',ascending=False).head(10)
def cos_sim(text,model):
    # compute similarity
    doc_sim = (
        1-cdist(
            document_embeddings_df.values,
            [model.wv[text]],
            'cosine'
        )
    )
    # convert result to a date frame
    document_sim_df = (
        pd.DataFrame(doc_sim, columns=["cos_sim"])
        .assign(document_id=list(document_embeddings_df.index))
    )
    # sort from most similar to least
    document_sim_df = document_sim_df.sort_values("cos_sim", ascending=False)
    
    # perform left-join to get information about the documents
    doc_sim_meta_df = document_sim_df.merge(all_data,
                      how='left',
                     left_on='document_id',
                     right_on='paper_id')
    return doc_sim_meta_df
def majority_voting(text,model):
    # choose top 100
    doc_sim_meta_dfs = [cos_sim(word,model).iloc[:100] for word in text.split()]
    return pd.merge(*doc_sim_meta_dfs,how = 'inner',on = 'document_id')
# replace empty text with empty strings
all_abstract = all_data.abstract.str.replace('\n\n', ' ')
ptn = r'\[[0-9]{1,2}\]'
all_abstract = all_abstract.str.replace(ptn,'').str.strip()
non_na = all_abstract.notna()
non_na_abstract = all_abstract[non_na]
non_na_abstract_paper_ids = all_data[non_na]['paper_id']
non_na_abstract_list = list(
    map(
        lambda x: word_tokenize(x), non_na_abstract.values
    ))
abst_documents = [TaggedDocument(doc, [non_na_abstract_paper_ids.values[i]]) for i, doc in enumerate(non_na_abstract_list)]
abst_model = Doc2Vec(abst_documents,vector_size = 300,window=2, min_count=1, workers=4)
document_dict_abst = {}
for idx, text_df in tqdm.tqdm(all_data[non_na][["paper_id", "abstract"]].iterrows()):
    document_dict_abst[text_df['paper_id']] = abst_model.docvecs[text_df['paper_id']]
abst_document_embeddings_df = pd.DataFrame.from_dict(document_dict_abst, orient="index")
abst_document_embeddings_df.head()
abst_doc_sim = (
    1 - cdist(
        abst_document_embeddings_df.values,
        [abst_model.wv['airborne']],
        'cosine'
    )
)
abst_doc_sim.shape
abst_document_sim_df = (
    pd.DataFrame(abst_doc_sim, columns=["cos_sim"])
    .assign(document_id=list(abst_document_embeddings_df.index))
)
abst_document_sim_df.sort_values('cos_sim',ascending=False)
majority_voting('non-pharmaceutical interventions',abst_model).loc[:100,'title_x']
cos_sim('airborne',abst_model).loc[:100,'title']
# replace empty text with empty strings
all_text = all_data.text.str.replace('\n\n', ' ')
ptn = r'\[[0-9]{1,2}\]'
all_text = all_text.str.replace(ptn,'').str.strip()
non_na_text_list = list(
    map(
        lambda x: word_tokenize(x), all_text.values
    ))
del all_text
gc.collect()
text_documents = [TaggedDocument(doc, [all_data.loc[i,'paper_id']]) for i, doc in enumerate(non_na_abstract_list)]
text_model = Doc2Vec(text_documents,vector_size = 300,window=2, min_count=1, workers=4)