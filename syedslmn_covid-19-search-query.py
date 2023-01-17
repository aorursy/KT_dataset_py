!pip install rank_bm25

import os
import json
import re
import pandas as pd
import spacy

from spacy.lang.en import English
from rank_bm25 import BM25Okapi
DOCUMENTS_PATH = '/kaggle/input/CORD-19-research-challenge'
ARTICLES_PATH = '/kaggle/working/documents.csv'
COVID_19_TERMS = [
    'covid',
    'covid 19',
    'covid 2019',
    'corona',
    'corona 19',
    'corona 2019',
    'coronavirus',
    'coronavirus 19',
    'coronavirus 2019',
    'wuhan virus',
    'china virus',
    'respiratory'
]
def extract_text(text_arr):
    txts = []
    return txts.append(x['text'] for x in text_arr)
def parse_documents():
    articles = []
    for dirpath, subdirs, files in os.walk(DOCUMENTS_PATH):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(dirpath, file)
                f = open(file_path)
                json_data = json.load(f)
                paper_id = json_data['paper_id']
                title = json_data['metadata']['title']
                articles_sub = []
                for x in json_data.get('abstract', []):
                    articles_sub.append(x['text'])
                for x in json_data.get('body_text', []):
                    articles_sub.append(x['text'])
                for sub in articles_sub:
                    full_txt = sub
                    actual_txt = sub
                    full_txt = full_txt.lower()
                    full_txt = re.sub('[^a-zA-Z0-9.\n]', ' ', full_txt)
                    full_txt = re.sub('http\S+|www.\S+', ' ', full_txt)
                    full_txt = re.sub('\t', ' ', full_txt)
                    full_txt = re.sub('\s+', ' ', full_txt)

                    res = [term for term in COVID_19_TERMS if(term in full_txt)] 
                    if bool(res):
                        articles.append([actual_txt, full_txt])
    data_df = pd.DataFrame(articles, columns=['actual_txt', 'processed_txt'])
    data_df.to_csv(ARTICLES_PATH)
parse_documents()
data_list = []
for chunk in pd.read_csv(ARTICLES_PATH, chunksize=2000):
    data_list.append(chunk)
data_df = pd.concat(data_list, axis= 0)

del data_list
data_df.head(10)
parser = English()
words = []
sents = []
i = 0
for index, row in data_df.iterrows():
    words.append([])
    sents.append(row['actual_txt'])
    doc = parser(row['processed_txt'])
    for token in doc:
        if not token.is_stop: # and token.tag_ in ['JJ', 'NN', 'NNP']:
            words[i].append(token.text)
    i = i + 1
bm25 = BM25Okapi(words)
bm25
def identify_docs(keywords, sents):
    scores = bm25.get_scores(keywords)
    sorted_scored = sorted(scores, reverse=True)
    identified_docs = bm25.get_top_n(keywords, sents, n=10)
    return identified_docs
queries = [
    'What is known about transmission, incubation, and environmental stability?',
    'What do we know about COVID-19 risk factors?',
    'What do we know about virus genetics, origin, and evolution?',
    'What do we know about vaccines and therapeutics?',
    'What has been published about medical care?',
    'What do we know about non-pharmaceutical interventions?',
    'What do we know about diagnostics and surveillance?',
    'Sample task with sample submission',
    'What has been published about ethical and social science considerations?',
    'What has been published about information sharing and inter-sectoral collaboration?'
]
nlp = spacy.load('en_core_web_sm')
keywords = ['transmission', 'incubation', 'environment', 'stability']
qry_keywords = []
for qry in queries:
    keywords = []
    qry = qry.lower()
    qry = re.sub('[^a-zA-Z0-9\n]', ' ', qry)
    qry = re.sub('http\S+|www.\S+', ' ', qry)
    qry = re.sub('\t', ' ', qry)
    qry = re.sub('\s+', ' ', qry)
    doc = nlp(qry)
    for token in doc:
        if not token.is_stop and token.pos_ in ['ADJ', 'NOUN', 'PROPN'] and token.text.strip():
            keywords.append(token.text)
            if (token.text != token.lemma_):
                keywords.append(token.lemma_)
    qry_keywords.append(keywords + COVID_19_TERMS)
for idx, qry in enumerate(queries):
    print()
    print('======', qry, '======')
    identified_docs = identify_docs(qry_keywords[idx], sents)
    for d in identified_docs:
        print('----', d)
