#first install the library that would help us use BERT in an easy to use interface

#https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers

!pip install -U sentence-transformers
#install the kaggle data to google colab

#https://github.com/Kaggle/kaggle-api#api-credentials

!pip install kaggle

import os

!cp "/content/kaggle.json" /root/.kaggle

!kaggle datasets download -d allen-institute-for-ai/CORD-19-research-challenge

!unzip  CORD-19-research-challenge.zip -d /content/CORD-19-research-challenge
import glob

import json

import pandas as pd

from tqdm import tqdm

root_path = '/content/CORD-19-research-challenge/'

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
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
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

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

    

    # add the journal information

    dict_['journal'].append(meta_data['journal'].values[0])

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])

df_covid.head()
df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)

df_covid['abstract'].describe(include='all')
df_covid['body_text'].describe(include='all')
df_covid.head()
df_covid.describe()
df_covid.dropna(inplace=True)

df_covid.info()
df_covid = df_covid.head(12500)
import re



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
def lower_case(input_str):

    input_str = input_str.lower()

    return input_str



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))
df_covid.head(4)
df_covid.to_csv("/content/drive/My Drive/BertSentenceSimilarity/Data/covid.csv")
df_covid_test = pd.read_csv("/content/drive/My Drive/BertSentenceSimilarity/Data/covid.csv")

text = df_covid_test.drop(["authors", "journal", "Unnamed: 0"], axis=1)

text.head(5)
text_dict = text.to_dict()

len_text = len(text_dict["paper_id"])
paper_id_list  = []

body_text_list = []



title_list = []

abstract_list = []

abstract_summary_list = []

for i in tqdm(range(0,len_text)):

  paper_id = text_dict["paper_id"][i]

  body_text = text_dict["body_text"][i].split("\n")

  title = text_dict["title"][i]

  abstract = text_dict["abstract"][i]

  abstract_summary = text_dict["abstract_summary"][i]

  for b in body_text:

    paper_id_list.append(paper_id)

    body_text_list.append(b)

    title_list.append(title)

    abstract_list.append(abstract)

    abstract_summary_list.append(abstract_summary)
df_sentences = pd.DataFrame({"paper_id":paper_id_list},index=body_text_list)

df_sentences.to_csv("/content/drive/My Drive/BertSentenceSimilarity/Data/covid_sentences.csv")

df_sentences.head()
df_sentences = pd.DataFrame({"paper_id":paper_id_list,"title":title_list,"abstract":abstract_list,"abstract_summary":abstract_summary_list},index=body_text_list)

df_sentences.to_csv("/content/drive/My Drive/BertSentenceSimilarity/Data/covid_sentences_Full.csv")

df_sentences.head()
import pandas as pd

from tqdm import tqdm



df_sentences = pd.read_csv("/content/drive/My Drive/BertSentenceSimilarity/Data/covid_sentences.csv")

df_sentences = df_sentences.set_index("Unnamed: 0")
df_sentences.head()
df_sentences = df_sentences["paper_id"].to_dict()

df_sentences_list = list(df_sentences.keys())

len(df_sentences_list)
list(df_sentences.keys())[:5]
df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]
import pandas as pd

df = pd.read_csv("/content/drive/My Drive/BertSentenceSimilarity/Data/covid_sentences_Full.csv", index_col=0)

df.head()
#https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.py

"""

This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,

we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.

"""



from sentence_transformers import SentenceTransformer

import scipy.spatial

import pickle as pkl

embedder = SentenceTransformer('bert-base-nli-mean-tokens')



# Corpus with example sentences

corpus = df_sentences_list

#corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)

with open("/content/drive/My Drive/BertSentenceSimilarity/Pickles/corpus_embeddings.pkl" , "rb") as file_:

  corpus_embeddings = pkl.load(file_)



# Query sentences:

queries = ['What has been published about medical care?',

           'Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest',

           'Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually',

           'Resources to support skilled nursing facilities and long term care facilities.',

           'Mobilization of surge medical staff to address shortages in overwhelmed communities .',

           'Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies .']

query_embeddings = embedder.encode(queries,show_progress_bar=True)



# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity

closest_n = 5

print("\nTop 5 most similar sentences in corpus:")

for query, query_embedding in zip(queries, query_embeddings):

    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]



    results = zip(range(len(distances)), distances)

    results = sorted(results, key=lambda x: x[1])



    print("\n\n=========================================================")

    print("==========================Query==============================")

    print("===",query,"=====")

    print("=========================================================")





    for idx, distance in results[0:closest_n]:

        print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )

        print("Paragraph:   ", corpus[idx].strip(), "\n" )

        row_dict = df.loc[df.index== corpus[idx]].to_dict()

        print("paper_id:  " , row_dict["paper_id"][corpus[idx]] , "\n")

        print("Title:  " , row_dict["title"][corpus[idx]] , "\n")

        print("Abstract:  " , row_dict["abstract"][corpus[idx]] , "\n")

        print("Abstract_Summary:  " , row_dict["abstract_summary"][corpus[idx]] , "\n")

        print("-------------------------------------------")
#import pickle as pkl

#with open("/content/drive/My Drive/BertSentenceSimilarity/Pickles/corpus_embeddings.pkl" , "wb") as file_:

#  pkl.dump(corpus_embeddings,file_)