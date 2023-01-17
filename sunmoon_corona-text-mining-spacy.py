#import useful modules

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

import os

import gc

import json

from gensim.models import Word2Vec

import tabulate





import numpy as np

import pandas as pd

import nltk

import spacy

from tqdm.notebook import tqdm

import glob

import re

import string

from gensim.models import FastText

from itertools import chain

import logging

import sys

from gensim import utils, matutils 

from numpy import dot

from IPython.core.display import display, HTML

from IPython.display import display, Markdown, Latex

tqdm.pandas()



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nlp = spacy.load("en_core_web_sm",disable=['parser','ner'])

stopwords = spacy.lang.en.stop_words.STOP_WORDS

table = str.maketrans('', '', string.punctuation)

covid_synonyms = ['coronavirus-2019-ncov','covid','covid-19','corona','coronavirus','ncp']
# read the processed df from other kernel (this is done to save the time for redundant tasks and accelerate the experimentation)

full_df = pd.read_csv("/kaggle/input/covid-data-preprocessing/corona_research_articles.csv")
# read the trained word2vec model

full_df.fillna('',inplace=True)

similarity_model = FastText.load("/kaggle/input/covid-data-preprocessing/fasttex_similarity_model.model")

similarity_model.init_sims(replace=True)
def get_must_have_words(query):

    words = [z.lower().translate(table) for z in re.findall(r"[\w']+|[.,!?;]",query) if z.lower() not in stopwords and not z.isnumeric() and len(z)>=3]

    return words
#convert all sections to vector

doc_vec_full = np.concatenate(full_df['tokenized_text'].progress_apply(lambda x:matutils.unitvec(np.array([similarity_model.wv[word] for word in list(chain.from_iterable([y.split(" ") for y in x.split(".")]))]).mean(axis=0))).values).reshape(-1,100)
queries = ["Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery",

           "Prevalence of asymptomatic shedding and transmission (e.g., particularly children)",

           "Seasonality of transmission",

           "Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding)",

           "Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood)",

           "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic)",

           "Natural history of the virus and shedding of it from an infected person",

           "Implementation of diagnostics and products to improve clinical processes",

           "Disease models, including animal models for infection, disease and transmission",

           "Tools and studies to monitor phenotypic change and potential adaptation of the virus",

           "Immune response and immunity",

           "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",

           "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",

           "Role of the environment in transmission"]

query_dict = {index:queries[index] for index in range(len(queries))}
# get the broad level similarity of each query with the text

query_words_list = []

for index, query in query_dict.items():

    query_words = get_must_have_words(query)

    query_vec = matutils.unitvec(np.array([similarity_model.wv[word] for word in query_words]).mean(axis=0).reshape(-1,100))

    full_df['{}_query_similarity'.format(index)] = np.matmul(doc_vec_full,query_vec.reshape(-1,1))#full_df['tokenized_text'].progress_apply(lambda x: get_similarity(x,query_words_list))
corona_vec = matutils.unitvec(np.array([similarity_model[word] for word in covid_synonyms]).mean(axis=0).reshape(-1,100))

full_df["corona_similarity"] = np.matmul(doc_vec_full,corona_vec.reshape(-1,1))
for query_num in range(len(queries)):

    query = query_dict[query_num]

    query_words = get_must_have_words(query)

    query_vec = matutils.unitvec(np.array([similarity_model.wv[word] for word in query_words]).mean(axis=0).reshape(-1,100))

    query_df = full_df.loc[full_df["{}_query_similarity".format(query_num)]>0.8]

    query_df["weighted_similarity"] = query_df["{}_query_similarity".format(query_num)]*query_df["corona_similarity"]

    query_df = query_df.sort_values(by=["weighted_similarity".format(query_num)],ascending=False).head(500)

    full_text = ""

    display(Markdown('<font color=green>**Query: {}**</font>'.format(query)))

    sentences, scores, titles = [], [], []

    for index,row in query_df.reset_index().iterrows():

        title = row['title']

        paper_id = row['paper_id']

        section_code = row['section_code']

        section_name = row['section_name']

        section_text = row['section_text']

        tokenized_text = row['tokenized_text']

        para_vec_full = np.concatenate([matutils.unitvec(np.array([similarity_model.wv[word] for word in y.split(" ")]).mean(axis=0).reshape(-1,100)) for y in tokenized_text.split(".")]).reshape(-1,100)

        similarity_score_query = np.array([similarity_model.wv.wmdistance(y.split(" "),query_words) for y in tokenized_text.split(".")])#np.matmul(para_vec_full,query_vec.reshape(-1,1)).reshape(-1,)

        similarity_score_corona = np.array([similarity_model.wv.wmdistance(y.split(" "),covid_synonyms) for y in tokenized_text.split(".")])#np.matmul(para_vec_full,corona_vec.reshape(-1,1)).reshape(-1,)

        similarity_score = similarity_score_query*0.7 + similarity_score_corona*0.3 #np.multiply(similarity_score_query,similarity_score_corona)

        n_len = len(similarity_score[np.where(similarity_score<=1.0)])

        n_sentences = n_len#max(n_len,3)

        arg_sort = np.argsort(similarity_score)

        sentences = sentences + [section_text.split(".")[rank] for rank in arg_sort[:n_sentences]]

        scores = scores + list(similarity_score[arg_sort[:n_sentences]])

        titles = titles + [title]*len(similarity_score[arg_sort[:n_sentences]])

        sorted_para = ".".join([section_text.split(".")[rank] for rank in arg_sort[:n_sentences]])

        full_text = full_text + sorted_para



    args = np.argsort(np.array(scores))

    sorted_sentences = [(titles[index], sentences[index],scores[index]) for index in np.argsort(np.array(scores))]

    table = [[titles[index], sentences[index], scores[index]] for index in np.argsort(np.array(scores))[:10]]

    display(HTML(tabulate.tabulate(table, ["Title","Relevant Text","dissimilarity (lower the better)"],tablefmt='html')))