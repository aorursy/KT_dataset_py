# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
!pip install pandas==1.0.3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import json
import functools
import time
# Any results you write to the current directory are saved as output.
df_clean_metadata = pd.read_csv('../input/coronawhy/clean_metadata.csv', index_col=0)
df_clean_metadata['publish_time'] = pd.to_datetime(df_clean_metadata['publish_time'])
df_citations = pd.read_csv('../input/coronawhy/cord19_MicrosoftAcademic_citation_metadata.csv')
df_all_metadata = df_clean_metadata.merge(df_citations, on='cord_uid', how='left')
df_all_metadata = df_all_metadata.set_index('sha')
df_all_metadata.head()
text_columns_to_keep = ['section', 'sentence', 'lemma', 'sentence_id']
method_words = ['methods', 'treatment']
result_words = ['results', 'statistics']
coronavirus_words = [
    'severe acute respiratory syndrome',
    'sars-cov',
    'sars-like',
    'middle east respiratory syndrome',
    'mers-cov',
    'mers-like',
    'covid-19',
    'sars-cov-2',
    '2019-ncov',
    'sars-2',
    'sarscov-2',
    'novel coronavirus',
    'corona virus',
    'coronaviruses',
    'coronaviruse',
    'sars',
    'mers',
    'covid19',
    'covid 19']
pre_existing_condition_words = [
    'chronic',
    'pre-existing',
    'persistent',
    'long-term',
    'genetic',
    'disability',
    'handicap',
    'disorder',
    'hereditary',
    'inborn errors',
    'defect',
    'weak immunity',
    'infirm']
ngrams = [
    'aortic valve stenosis',
    'rheumatic heart disease',
    'complete atrioventricular block',
    'congestive heart failure',
    'sick sinus syndrome',
    'junctional premature complex',
    'pulmonary heart disease',
    'mitral valve regurgitation',
    'supraventricular premature beats',
    'supraventricular tachycardia',
    'pulmonic valve stenosis',
    'primary pulmonary hypertension',
    'tricuspid valve stenosis',
    'ventricular premature complex',
    'left heart failure',
    'hypertensive renal disease',
    'primary pulmonary hypertension',
    'acute myocardial infarction',
    'coronary atherosclerosis',
    'ischemic chest pain',
    'primary dilated cardiomyopathy',
    'hypertensive disorder',
    'cardiac arrest',
    'myocardial ischemia',
    'myocardial disease',
    'heart murmur']
def generate_ngram_data_from_paper(ngram, pre_existing_word, paper_id, coronavirus_words, method_words, result_words, df_paper, metadata):
    df_matching_ngram = df_paper[df_paper['sentence'].str.lower().str.contains(ngram)]
    df_matching_pre_existing = df_paper[df_paper['sentence'].str.lower().str.contains(pre_existing_word)]
    
    matching_sentences_in_method = len(df_matching_ngram[df_matching_ngram['section'].isin(method_words)]['sentence'])
    matching_sentences_in_result = len(df_matching_ngram[df_matching_ngram['section'].isin(result_words)]['sentence'])
    abstract = metadata['abstract']
    if not isinstance(abstract, str):
        abstract = ''
    abstract = abstract.lower()

    if ngram in abstract and matching_sentences_in_result > 0 and len(df_matching_pre_existing):
        return None
    
    
    contains_covid = df_paper['sentence'].str.lower().str.contains('|'.join(coronavirus_words), regex=True).any()
    # how would you get correlation?
    publish_time = metadata['publish_time']
    title = metadata['title']
    url = metadata['url']
    authors = metadata['authors']
    num_citations = metadata['CitationCount']
    source = metadata['journal']
    num_heart_keywords = df_paper['sentence'].str.lower().str.count(ngram).sum()        
    full_text_method = df_paper[df_paper['section'].isin(method_words)]['sentence'].str.cat()
    full_text_result = df_paper[df_paper['section'].isin(result_words)]['sentence'].str.cat()

    data = {
        'ngram': ngram,
        'pre_existing_word': pre_existing_word,
        'correlation': None,
        'publish_time': publish_time,
        'paper_id': paper_id,
        'paper_title': title,
        'url': url,
        'matching_sentences_in_method': matching_sentences_in_method,
        'matching_sentences_in_result': matching_sentences_in_result,
        'authors': authors,
        'num_citations': num_citations,
        'num_heart_keywords': num_heart_keywords,
        'contains_covid': contains_covid,
        'paper_source': source,
        'full_text_method': full_text_method,
        'full_text_result': full_text_result
    }
    return data
def process_tsv(filepath, df_all_metadata, paper_ids, ngrams, pre_existing_condition_words, coronavirus_words, method_words, result_words):
    ngram_papers = []
    df = pd.read_pickle(filepath, 'gzip')
 
    for paper_id, df_paper in df.groupby('paper_id'):
        if paper_id not in paper_ids:
            continue
        df_paper = df_paper.loc[df_paper['sentence'].dropna().index]
        if df_paper['sentence'].str.lower().str.contains('|'.join(ngrams), regex=True).sum() == 0:
            continue
        if df_paper['sentence'].str.lower().str.contains('|'.join(pre_existing_condition_words), regex=True).sum() == 0:
            continue
        metadata = df_all_metadata.loc[paper_id]
        for ngram in ngrams:
            for pre_existing_word in pre_existing_condition_words:
                data = generate_ngram_data_from_paper(ngram, pre_existing_word, paper_id, coronavirus_words, method_words, result_words, df_paper, metadata)
            if data is None:
                continue
            ngram_papers.append(data)
    return ngram_papers
input_directory = '../input/coronawhy/v6_text/v6_text/'
ngram_papers = []
paper_ids = set(df_all_metadata.index)

for filename in os.listdir(input_directory):
    print(filename)
    filepath = os.path.join(input_directory, filename)
    new_ngram_papers = process_tsv(filepath, df_all_metadata, paper_ids, ngrams, pre_existing_condition_words, coronavirus_words, method_words, result_words)
    ngram_papers += new_ngram_papers

df_final = pd.DataFrame(ngram_papers)
df_final
df_final.to_csv('ngram_risk_factor_heart_disease.csv')
