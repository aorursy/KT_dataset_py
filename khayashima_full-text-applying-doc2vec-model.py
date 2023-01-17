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
!pip install swifter

#!pip install scispacy

#!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
# Progress bar

import tqdm



# Word2Vec

from gensim.models.word2vec import Word2Vec

from gensim.models.doc2vec import Doc2Vec,TaggedDocument



from nltk.tokenize import word_tokenize,sent_tokenize 

from scipy.spatial.distance import cdist,cosine

import gc

import swifter

import spacy
#nlp = spacy.load("en_core_web_sm")

#tokenizer = nlp.Defaults.create_tokenizer(nlp)
all_data = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv").drop(columns = ['bibliography','raw_bibliography','raw_authors'])

"""

biorxiv_clean = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv").drop(columns = ['bibliography','raw_bibliography','raw_authors'])

clean_comm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv").drop(columns = ['bibliography','raw_bibliography','raw_authors'])

clean_noncomm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv").drop(columns = ['bibliography','raw_bibliography','raw_authors'])

clean_pmc = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv").drop(columns = ['bibliography','raw_bibliography','raw_authors'])



all_data = pd.concat([biorxiv_clean, clean_comm_use, clean_noncomm_use, clean_pmc]).reset_index(drop=True).drop_duplicates()

del biorxiv_clean,clean_comm_use,clean_noncomm_use,clean_pmc

gc.collect()

"""

all_data.head()
print("Number of Rows in Table: %i" % len(all_data))

print("Number of Titles: %i " % all_data['title'].count())

print("Number of Abstracts: %i " % all_data['abstract'].count())

print("Number of Texts: %i " % all_data['text'].count())
def cos_sim(text,df,model):

    # compute similarity

    doc_sim = (

        1-cdist(

            df.values,

            [model.wv[text]],

            'cosine'

        )

    )

    # convert result to a date frame

    document_sim_df = (

        pd.DataFrame(doc_sim, columns=["cos_sim"])

        .assign(document_id=list(df.index))

    )

    # sort from most similar to least

    document_sim_df = document_sim_df.sort_values("cos_sim", ascending=False)

    

    # perform left-join to get information about the documents

    doc_sim_meta_df = document_sim_df.merge(all_data,

                      how='left',

                     left_on='document_id',

                     right_on='paper_id')

    return doc_sim_meta_df
def majority_voting(text,df,model):

    # choose top 100

    doc_sim_meta_dfs = [cos_sim(word,df,model).iloc[:100] for word in text.split()]

    return pd.merge(*doc_sim_meta_dfs,how = 'inner',on = 'document_id')
# replace empty text with empty strings

ptn = r'\[[0-9]{1,2}\]'
def tokenize_and_tag(x):

    return TaggedDocument(word_tokenize(x['text'].replace('\n\n', ' ').replace(ptn,'').strip()),[x['paper_id']])
#text_documents = list(map( lambda x: tokenizer(x),all_data.text.str.replace('\n\n', ' ').replace(ptn,'').str.strip()))

text_documents = all_data.swifter.apply(tokenize_and_tag,axis = 1)
#text_documents = [TaggedDocument(doc, [all_data.loc[i,'paper_id']]) for i, doc in enumerate(text_documents)]
text_model = Doc2Vec(text_documents,vector_size = 200,window=10, min_count=3, workers=4)

text_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
document_dict_text = {}

for idx, text_df in tqdm.tqdm(all_data[["paper_id", "text"]].iterrows()):

    document_dict_text[text_df['paper_id']] = text_model.docvecs[text_df['paper_id']]
text_document_embeddings_df = pd.DataFrame.from_dict(document_dict_text, orient="index")

text_document_embeddings_df.head()
npi_papers = majority_voting('non-pharmaceutical interventions',text_document_embeddings_df,text_model).loc[:100,['text_x','title_x']]
# identify paragraphs for each document

# each line breaks will give us paragraph 

npi_papers['text_x'] = npi_papers['text_x'].replace('\n\n', ' ').str.split('\n')

npi_tagged_docs = []

paragraph_table = []

for i,row in npi_papers.iterrows():

    title = row['title_x']

    npi_tagged_docs += [TaggedDocument(word_tokenize(text),[f"{title} - {j}"]) for j,text in enumerate(row['text_x']) if len(text) > 1]

    paragraph_table += [[title,j,text] for j,text in enumerate(row['text_x']) if len(text) > 1]

paragraph_df = pd.DataFrame(data = paragraph_table,columns = ['title','paragraph_id','text'])
paragraph_model = Doc2Vec(npi_tagged_docs,vector_size = 150,window=4, min_count=3, workers=4)

paragraph_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
def get_paragraph_model_dic(x,text):

    title = x[0]

    paragraph_id = x[1]

    doc_vec = paragraph_model.docvecs[f"{title} - {paragraph_id}"]

    cos_sim = 1 - cosine(

            doc_vec,

            paragraph_model.wv[text]

        )

    return cos_sim
paragraph_df['non_ph_cos_sim'] = paragraph_df.swifter.apply(get_paragraph_model_dic,axis=1,text = 'non-pharmaceutical')

paragraph_df['interv_cos_sim'] = paragraph_df.swifter.apply(get_paragraph_model_dic,axis=1,text = 'intervention')

top_paragraphs = pd.merge(paragraph_df.sort_values(by = 'non_ph_cos_sim',ascending = False).iloc[:200],

        paragraph_df.sort_values(by = 'interv_cos_sim',ascending = False).iloc[:200],

         on = ['title','paragraph_id'],how = 'inner')
top_paragraphs
top_paragraphs.to_csv('top_paragraphs.csv',index = False)
# predict subtasks 

paragraph_model.infer_vector(word_tokenize("Methods to control the spread in communities, barriers to compliance and how these vary among different populations."))