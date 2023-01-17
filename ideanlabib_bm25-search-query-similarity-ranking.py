!pip install rank_bm25 nltk
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path, PurePath

import requests

from requests.exceptions import HTTPError, ConnectionError

from ipywidgets import interact

import ipywidgets as widgets

from rank_bm25 import BM25Okapi

import nltk

from nltk.corpus import stopwords

nltk.download("punkt")

import re

import os

import json

import glob

import sys

from tqdm import tqdm

from rank_bm25 import BM25Okapi
sys.path.insert(0, "../")



DATA_PATH = '/kaggle/input/CORD-19-research-challenge'



# Walks all subdirectories in a directory, and their files. 

# Opens all json files we deem relevant, and append them to

# a list that can be used as the "data" argument in a call to 

# pd.DataFrame.

def gather_jsons(dirName):

    

    # Get the list of all files in directory tree at given path

    # include only json with encoded id (40-character SHA hash)

    # Only length of filename is checked, but this should be sufficient

    # given the task.

    

    listOfFiles = list()

    for (dirpath, dirnames, filenames) in os.walk(dirName):

        listOfFiles += [os.path.join(dirpath, file) for file in filenames

                        if file.endswith("json")

                        and len(file) == 45]

    jsons = []

    

    print(str(len(listOfFiles)) + " jsons found! Attempting to gather.")

    

    for file in tqdm(listOfFiles):

        with open(file) as json_file:

            jsons.append(json.load(json_file))

    return jsons

        

        

# Returns a dictionary object that's easy to parse in pandas.

def extract_from_json(json):

    

    # For text mining purposes, we're only interested in 4 columns:

    # abstract, paper_id (for ease of indexing), title, and body text.

    # In this particular dataset, some abstracts have multiple sections,

    # with ["abstract"][1] or later representing keywords or extra info. 

    # We only want to keep [0]["text"] in these cases. 

    if len(json["abstract"]) > 0:

        json_dict = {

            "_id": json["paper_id"],

            "title": json["metadata"]["title"],

            "abstract": json["abstract"],

            "text": " ".join([i["text"] for i in json["body_text"]])

        }

        

    # Else, ["abstract"] isn't a list and we can just grab the full text.

    else:

        json_dict = {

            "_id": json["paper_id"],

            "title": json["metadata"]["title"],

            "abstract": json["abstract"],

            "text": " ".join([i["text"] for i in json["body_text"]])

        }



    return json_dict



# Combines gather_jsons and extract_from_json to create a

# pandas DataFrame object.

def gather_data(dirName):

    return(pd.DataFrame(data=[extract_from_json(json) for json in gather_jsons(dirName)]))



corona_df = gather_data(f"{DATA_PATH}")

corona_df['abstract'] = corona_df.abstract.apply(lambda x: ' '.join([r['text'] for r in x]))

corona_df.to_csv("covid_data_full.csv", index=False)
english_stopwords = list(set(stopwords.words('english')))



def strip_characters(text):

    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)

    t = re.sub('/', ' ', t)

    t = t.replace("'",'')

    return t



def clean(text):

    t = text.lower()

    t = strip_characters(t)

    return t



def tokenize(text):

    words = nltk.word_tokenize(text)

    return list(set([word for word in words 

                     if len(word) > 1

                     and not word in english_stopwords

                     and not (word.isnumeric() and len(word) is not 4)

                     and (not word.isnumeric() or word.isalpha())] )

               )



def preprocess(text):

    t = clean(text)

    tokens = tokenize(t)

    return tokens



class SearchResults:

    

    def __init__(self, 

                 data: pd.DataFrame,

                 columns = None):

        self.results = data

        if columns:

            self.results = self.results[columns]

            

    def __getitem__(self, item):

        return Paper(self.results.loc[item])

    

    def __len__(self):

        return len(self.results)

        

    def _repr_html_(self):

        return self.results._repr_html_()



SEARCH_DISPLAY_COLUMNS = ['_id', 'title', 'abstract', 'text']



class WordTokenIndex:

    

    def __init__(self, 

                 corpus: pd.DataFrame, 

                 columns=SEARCH_DISPLAY_COLUMNS):

        self.corpus = corpus

        raw_search_str = self.corpus.abstract.fillna('') + ' ' + self.corpus.title.fillna('') + ' ' + self.corpus.text.fillna('')

        self.index = raw_search_str.apply(preprocess).to_frame()

        self.index.columns = ['terms']

        self.index.index = self.corpus.index

        self.columns = columns

    

    def search(self, search_string):

        search_terms = preprocess(search_string)

        result_index = self.index.terms.apply(lambda terms: any(i in terms for i in search_terms))

        results = self.corpus[result_index].copy().reset_index().rename(columns={'index':'paper'})

        return SearchResults(results, self.columns + ['paper'])
class RankBM25Index(WordTokenIndex):

    

    def __init__(self, corpus: pd.DataFrame, columns=SEARCH_DISPLAY_COLUMNS):

        super().__init__(corpus, columns)

        self.bm25 = BM25Okapi(self.index.terms.tolist())

        

    def search(self, search_string, n=4):

        search_terms = preprocess(search_string)

        doc_scores = self.bm25.get_scores(search_terms)

        ind = np.argsort(doc_scores)[::-1][:n]

        results = self.corpus.iloc[ind][self.columns]

        results['Score'] = doc_scores[ind]

        results['orig_ind'] = ind

        results = results[results.Score > 0]

        return SearchResults(results.reset_index(), self.columns + ['Score', 'orig_ind'])

bm25_index = RankBM25Index(corona_df)
keywords = ['sars-cov', 'sars', 'coronavirus', 'ace2', 'coronaviruses', 'ncov', 'covid-19', 'wuhan', 'spike', 'sars-cov-2']
results = None

added = []

for word in keywords:

    word_result = bm25_index.search(word, n=100).results

    if results is None:

        results = word_result

        added += [r.orig_ind for i, r in word_result.iterrows()]

        continue

    for i, r in word_result.iterrows():

        if r.orig_ind not in added:

            results = results.append(r)

            added.append(r.orig_ind)

df = results.sort_values(by='Score', ascending=False)

df.reset_index(drop=True, inplace=True)
# !tar -xvf /kaggle/input/spacy-covid19/covid-19-en_lg.tar.xz
!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz

import scispacy

import spacy

import en_core_sci_lg

nlp = en_core_sci_lg.load()

# nlp = spacy.load("covid-19-en_lg/")

nlp.max_length=2000000
from tqdm import tqdm



vector_list = []

for i in tqdm(df.index):

    doc = nlp(df.iloc[i].text)

    sents = [sent for sent in doc.sents]

    vecs = [sent.vector for sent in sents]

    for j in range(len(sents)):

        vector_list.append(

            {"_id": df.iloc[i]._id, 

             "score": df.iloc[i]['Score'],

             "sentence": j, 

             "vector": vecs[j], 

             "start_span": sents[j].start_char,

             "end_span": sents[j].end_char})

vector_df = pd.DataFrame(data=vector_list)
queries = """Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.

Prevalence of asymptomatic shedding and transmission. 

Prevalence of asymptomatic shedding and transmission in children, infants, and young people.

Seasonality of transmission of the virus.

Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic or hydrophobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).

Persistence and stability on a multitude of substrates and sources (nasal discharge, sputum, urine, fecal matter, blood, bodily fluids and secretions).

Persistence of virus on surfaces of different materials (copper, stainless steel, plastic).

Natural history of the virus and shedding of it from an infected person.

Implementation of diagnostics and products to improve clinical processes.

Disease models, including animal models for infection, disease and transmission.

Tools and studies to monitor phenotypic change and potential adaptation of the virus.

Immune response and immunity to the virus.

Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings.

Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings.

Role of the environment in transmission."""

queries = queries.splitlines()

queries_df = pd.DataFrame(data=[{"query":query} for query in queries])
query_vector_list = []

for i in tqdm(range(len(queries))):

    doc = nlp(queries[i])

    vec = doc.vector

    query_vector_list.append({"_id": f"query_{i}", "vector": vec})

    

query_vector_df = pd.DataFrame(data=query_vector_list)

query_vector_df.to_csv("query_vecs.csv",index=False)
from scipy.spatial import distance

RATIO = 0.9

distances = distance.cdist([value for value in query_vector_df["vector"]], [value for value in vector_df["vector"].values], "cosine")

w2v_searchable_df = vector_df.drop(columns=["vector"])

# Create a column with cosine distances for each query vs the sentence

for i in range(len(queries)):

    #w2v_searchable_df[f"query_{i}_distance"] = 1 - (np.power((1 - distances[i]), RATIO) * w2v_searchable_df['score'])

    w2v_searchable_df[f"query_{i}_distance"] = RATIO * (1 - distances[i]) + (1-RATIO) * w2v_searchable_df['score']

w2v_searchable_df.to_csv("covid_w2v_searchable.csv", index=False)
for i in range(len(queries)):

    columnName = f"query_{i}_distance"

    context = w2v_searchable_df.sort_values(by=columnName, ascending=False)[["_id","start_span","end_span"]][:20]

    ix = context["_id"].to_list()

    spans1 = context["start_span"].to_list()

    spans2 = context["end_span"].to_list()

    print("Question: " + queries[i] + "\n")

    for j in range(len(context.index)):

        score = df[df["_id"] == ix[j]].iloc[0]['Score']

        print(f"Rank {j+1} (Relevance Score {score}): " + str(df[df["_id"] == ix[j]].iloc[0]["text"])[spans1[j]:spans2[j]] + "\n")