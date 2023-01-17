!pip install -U spacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
from __future__ import unicode_literals, print_function
from pathlib import Path
from spacy.util import minibatch, compounding
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import itertools
import json
import nltk.data
import numpy as np
import os
import pandas as pd
import random
import spacy
# CONFIG

# Data
DIR_DATA_INPUT = os.path.join('/kaggle', 'input', 'CORD-19-research-challenge')
DIR_BIORXIV = os.path.join(DIR_DATA_INPUT, 'biorxiv_medrxiv', 'biorxiv_medrxiv', 'pdf_json')
DIR_COMM = os.path.join(DIR_DATA_INPUT, 'comm_use_subset', 'comm_use_subset', 'pdf_json')
DIR_CUSTOM = os.path.join(DIR_DATA_INPUT, 'custom_license', 'custom_license', 'pdf_json')
DIR_NONCUSTOM = os.path.join(DIR_DATA_INPUT, 'noncomm_use_subset', 'noncomm_use_subset', 'pdf_json')

DIR_DATA_OUTPUT = os.path.join('/kaggle', 'working')
PATH_AGG_JSON = os.path.join(DIR_DATA_OUTPUT, 'agg_data.json')
def extract_jsons_to_list(folder):
    """
    Extracting 4 fields ('abstract', 'text', 'paper_id', 'title') from orginal Json file
    :folder String, to location with Jsons
    :return: Lists, with selected params
    """
    results = []

    files = os.listdir(folder)
    for filename in tqdm(files, f'parsing {folder}'):
        json_file = os.path.join(folder, filename)
        file = json.load(open(json_file, 'rb'))
        agg_abstract_file = ' '.join(
            [abstract['text'] for abstract in file['abstract']])
        text = ' '.join(
            [text['text'] for text in file['body_text']])
        results.append({
            'abstract': agg_abstract_file,
            'text': text,
            'paper_id': file['paper_id'], 
            'title': file['metadata']['title']
        })

    return results


def save_json(file_to_save, path_to_save):
    """
    Save in relevant Json format
    :file_to_save DataFrame, file to save
    :path_to_save String, lacation to save a file
    """
    df = pd.DataFrame(file_to_save)
    
    df['json_output'] = df.apply(lambda x: {
        'text': x.text, "meta":{'paper_id':x.paper_id, 'title': x.title}
    }, axis=1)
    df['json_output'].to_json(path_to_save, orient='records', lines=True)
    

def filtr_covid_and_risk_factor(file_to_save, path_to_save):
    """
    List filtering in abstact and text (filters: 'COVID-19' or 'SARS-CoV-2')
    :file_to_save List, file to save
    :path_to_save String, lacation to save a file
    :return: DataFrame, valid data
    """
    df = pd.DataFrame(file_to_save)
    mask = df['abstract'].str.contains('COVID-19') | df['text'].str.contains('COVID-19') \
     | df['abstract'].str.contains('SARS-CoV-2') | df['text'].str.contains('SARS-CoV-2')
    
    abstracts = text_2_sentance(df[mask], 'abstract')
    text = text_2_sentance(df[mask], 'text')
    abstracts.extend(text)

    save_json(abstracts, path_to_save)
    
    return df


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def text_2_sentance(df, column):
    """
    Save 3 senctance before and after sentance which contains `risk factor` expression
    :df DataFrame, with text data
    :column String, column name to process
    :return: List, valid sentance
    """
    df['sentances'] = df.apply(lambda x: tokenizer.tokenize(x[column]), axis = 1)
    
    valid_sentance = []
    for _, row in tqdm(df.iterrows()):
        sentance_range = set()
        for index, singiel_sentance in enumerate(row['sentances']):
            if 'risk factor' in singiel_sentance.lower():
                sentance_range.update(
                    range(index-3, index+4))
        for valid_index in sentance_range:
            if valid_index >=0 and valid_index < len(row['sentances']):
                valid_sentance.append({
                    'text': row['sentances'][valid_index],
                    'paper_id': row['paper_id'], 
                    'title': row['title']
                })
                
    return valid_sentance

# Generate Json for Marek

bio = extract_jsons_to_list(DIR_BIORXIV)
comm = extract_jsons_to_list(DIR_COMM)
cus = extract_jsons_to_list(DIR_CUSTOM)
non = extract_jsons_to_list(DIR_NONCUSTOM)

list_agg = bio + comm + cus + non
results = filtr_covid_and_risk_factor(list_agg, PATH_AGG_JSON)

!wget https://raw.githubusercontent.com/chopeen/CORD-19/master/data/annotated/cord_19_rf_sentences_merged.json
!ls -1
new_list = []
file = json.load(open('cord_19_rf_sentences_merged.json', 'rb'))

df = pd.DataFrame(file)

X_train, X_test = train_test_split(
    df, test_size=0.2, random_state=42)

X_train.to_json('train_abstract_teach.json', orient='records')
X_test.to_json('test_abstract_teach.json', orient='records')
!spacy train en models/ train_abstract_teach.json test_abstract_teach.json --pipeline ner --base-model en_core_sci_lg  --replace-components