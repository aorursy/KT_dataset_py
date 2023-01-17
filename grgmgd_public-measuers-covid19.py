# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import nltk

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
dataset_directory = '/kaggle/input/CORD-19-research-challenge/'
with open(f'{dataset_directory}/metadata.readme', 'r') as file:
    data = file.read()
    print(data)
metadata = pd.read_csv(f'{dataset_directory}/metadata.csv')
metadata.head()
corpus_files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        ifile = os.path.join(dirname, filename)
        if ifile.split(".")[-1] == "json":
            corpus_files.append(ifile)
            
len(corpus_files)
abstracts = []
for file in corpus_files:
    with open(file,'r')as f:
        doc = json.load(f)
    if 'abstract' not in doc:
        continue
    id = doc['paper_id'] 
    abstract = ''
    for item in doc['abstract']:
        abstract = abstract + item['text']
        
    abstracts.append({id: abstract})

abstracts[22]
with open("/kaggle/working/abstracts.json", 'w') as f:
    json.dump(abstracts, f)
with open("/kaggle/input/covid19-abstracts/abstracts.json", 'r') as f:
    abstracts = json.load(f)
abstracts[22]
!python -m spacy download en_core_web_lg
import spacy
from spacy.matcher import PhraseMatcher, Matcher

nlp = spacy.load("en_core_web_lg")

matcher = Matcher(nlp.vocab)
matcher.add("methods_medical_mathcer", None, 
            [{"LEMMA": {"IN": ["measures", "control"]}}, {"POS": "NOUN"}],
            [{"LEMMA": {"IN": ["effective", "prevention"]}}, {"POS": "NOUN"}],
           )

document = nlp(list(abstracts[22].values())[0])
matches = matcher(document)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]  # Get string representation
    span = document[start:end]  # The matched span
    print(match_id, string_id, start, end, span.text)
    
if not matches:
    print("This document doesn't match the query")
query = nlp("prevention and control measures to halt virus spread in population")

def is_relevant(abstract): 
    document = nlp(abstract)
    matches = matcher(document)
    similarity = 0
    if matches:
        similarity = document.similarity(query)
    return matches, similarity

relevant = []            
for index in range(len(abstracts)):
    match, similarity = is_relevant(list(abstracts[index].values())[0])
    if match and similarity > 0.85:
        relevant.append({ index: {"matches": match, "score": similarity}})
print(len(relevant))
with open("/kaggle/working/relevant.json", 'w') as f:
    json.dump(relevant, f)
with open("/kaggle/input/covid19-abstracts/relevant.json", 'r') as f:
    relevant = json.load(f)
sorted_relevance = sorted(relevant, key=lambda x: list(x.values())[0]["score"], reverse=True)
relevant_abstarct_keys = [int(list(item.keys())[0]) for item in sorted_relevance[0:30]]
abstracts[relevant_abstarct_keys[2]]
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

extra_words = list(STOP_WORDS) + list(punctuation) + ['\n']

def summarizer(document):
    docx = nlp(document)

    all_words=[word.text for word in docx]
    Freq_word={}
    for w in all_words:
        w1=w.lower()
        if w1 not in extra_words and w1.isalpha():
            if w1 in Freq_word.keys():
                Freq_word[w1] += 1
            else:
                Freq_word[w1] = 1

    for word in Freq_word.keys():
        Freq_word[word] = (Freq_word[word]/max_freq[-1])

    sent_strength={}
    for sent in docx.sents:
        for word in sent :
            if word.text.lower() in Freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += Freq_word[word.text.lower()]
                else:
                    sent_strength[sent] = Freq_word[word.text.lower()]
            else: 
                continue

    top_sentences = (sorted(sent_strength.values())[::-1])
    top = int(0.35*len(top_sentences))
    top_sent = top_sentences[:top]

    summary=[]
    for sent,strength in sent_strength.items():
        if strength in top_sent:
            summary.append(sent)
        else:
            continue
    return ''.join(map(str, summary))
for index in relevant_abstarct_keys:
    summary = summarizer(list(abstracts[index].values())[0])
    print("abstract", index, ": ", summary, "\n")