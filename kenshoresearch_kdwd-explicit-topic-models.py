from collections import Counter

import itertools

import json

import os

import random

import re



import joblib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy.sparse

import seaborn as sns

from tqdm import tqdm



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
pd.set_option('max_colwidth', 160)

pd.set_option('display.max_colwidth', 60)
sns.set()

sns.set_context('notebook')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
NUM_KLAT_LINES = 5_343_564

KDWD_PATH = os.path.join(

    '/kaggle', 

    'input', 

    'kensho-derived-wikimedia-data'

)

SKIP_SECTION_NAMES = [

    'see also', 

    'external links', 

    'further reading', 

    'references', 

    'bibliography'

]
page_df = pd.read_csv(

    os.path.join(KDWD_PATH, "page.csv"),

    keep_default_na=False, # dont read the page title "NaN" as a null

) 

page_df.head()
class KdwdLinkAnnotatedText:

    

    def __init__(self, file_path):

        self.file_path = file_path

        self.num_lines = NUM_KLAT_LINES



    def __iter__(self):

        with open(self.file_path) as fp:

            for line in tqdm(fp, total=self.num_lines):

                page = json.loads(line)

                yield page 
file_path = os.path.join(KDWD_PATH, "link_annotated_text.jsonl")

klat = KdwdLinkAnnotatedText(file_path)
out_links = Counter()

in_links = Counter()

for page in klat:

    for section in page['sections']:

        out_links[page['page_id']] += len(section['target_page_ids'])

        in_links.update(section['target_page_ids'])
page_df = pd.merge(

    page_df, 

    pd.DataFrame(in_links.most_common(), columns=['page_id', 'in_links']), 

    how='left').fillna(0.0)
page_df = pd.merge(

    page_df, 

    pd.DataFrame(out_links.most_common(), columns=['page_id', 'out_links']), 

    how='left').fillna(0.0)
page_df
del out_links

del in_links
topic_defs = {

    'large': {

        'min_in_links': 50,

        'min_out_links': 20,

        'min_views': 100,

    },

    'medium': {

        'min_in_links': 500,

        'min_out_links': 20,

        'min_views': 100,

    },

    'small': {

        'min_in_links': 5000,

        'min_out_links': 20,

        'min_views': 100,

    },

}



model_defs = {

    'base_large': {

        'topics': 'large',

        'intros_only': False,

        'min_df': 3,

        'max_df': 0.98,

        'ngram_range': (1,1),

        'stop_words': ENGLISH_STOP_WORDS,

        'max_features': 1_000_000,

        'token_pattern': r"(?u)\b[^\d\W]{2,25}\b",

    },

    'base_medium': {

        'topics': 'medium',

        'intros_only': False,

        'min_df': 3,

        'max_df': 0.98,

        'ngram_range': (1,1),

        'stop_words': ENGLISH_STOP_WORDS,

        'max_features': 1_000_000,

        'token_pattern': r"(?u)\b[^\d\W]{2,25}\b",

    },

    'base_small': {

        'topics': 'small',

        'intros_only': False,

        'min_df': 3,

        'max_df': 0.98,

        'ngram_range': (1,1),

        'stop_words': ENGLISH_STOP_WORDS,

        'max_features': 1_000_000,

        'token_pattern': r"(?u)\b[^\d\W]{2,25}\b",

    },    

    'intros_large': {

        'topics': 'large',

        'intros_only': True,

        'min_df': 3,

        'max_df': 0.98,

        'ngram_range': (1,1),

        'stop_words': ENGLISH_STOP_WORDS,

        'max_features': 1_000_000,

        'token_pattern': r"(?u)\b[^\d\W]{2,25}\b",

    },

    'intros_medium': {

        'topics': 'medium',

        'intros_only': True,

        'min_df': 3,

        'max_df': 0.98,

        'ngram_range': (1,1),

        'stop_words': ENGLISH_STOP_WORDS,

        'max_features': 1_000_000,

        'token_pattern': r"(?u)\b[^\d\W]{2,25}\b",

    },

    'intros_small': {

        'topics': 'small',

        'intros_only': True,

        'min_df': 3,

        'max_df': 0.98,

        'ngram_range': (1,1),

        'stop_words': ENGLISH_STOP_WORDS,

        'max_features': 1_000_000,

        'token_pattern': r"(?u)\b[^\d\W]{2,25}\b",

    },

    'intros_small_12': {

        'topics': 'small',

        'intros_only': True,

        'min_df': 3,

        'max_df': 0.98,

        'ngram_range': (1,2),

        'stop_words': ENGLISH_STOP_WORDS,

        'max_features': 1_000_000,

        'token_pattern': r"(?u)\b[^\d\W]{2,25}\b",

    },

    'intros_small_13': {

        'topics': 'small',

        'intros_only': True,

        'min_df': 3,

        'max_df': 0.98,

        'ngram_range': (1,3),

        'stop_words': ENGLISH_STOP_WORDS,

        'max_features': 1_000_000,

        'token_pattern': r"(?u)\b[^\d\W]{2,25}\b",

    },

}
topic_dfs = {}

for model_name, model_params in model_defs.items():

    if model_params['topics'] in topic_dfs:

        continue

    else:

        topic_params = topic_defs[model_params['topics']]

        mask1 = page_df['in_links'] >= topic_params['min_in_links']

        mask2 = page_df['out_links'] >= topic_params['min_out_links']

        mask3 = page_df['views'] > topic_params['min_views']

        topic_dfs[model_params['topics']] = page_df[mask1 & mask2 & mask3].copy().reset_index(drop=True)
topic_dfs['small']
topic_dfs['medium']
topic_dfs['large']
def gen_text(klat, topic_df, intros_only):

    keep_page_ids = set(topic_df['page_id'].values)

    for page in klat:

        if page['page_id'] not in keep_page_ids:

            continue

        sections = [

            section for section in page['sections']

            if (section['name'] is not None) and (section['name'].lower() not in SKIP_SECTION_NAMES)]

        if intros_only:

            yield sections[0]['text']

        else:

            yield ' '.join([section['text'] for section in sections])
key = 'small'

next(iter(gen_text(klat, topic_dfs[key], intros_only=False)))[0:1000]
cvs = {}

for model_name, model_params in model_defs.items():

    cvs[model_name] = CountVectorizer(

        min_df=model_params['min_df'],

        max_df=model_params['max_df'],

        stop_words=model_params['stop_words'],

        ngram_range=model_params['ngram_range'],

        max_features=model_params['max_features'],

        token_pattern=model_params['token_pattern'],

    )
Xcvs = {}

for model_name, model_params in model_defs.items():

    topic_df = topic_dfs[model_params['topics']]

    Xcvs[model_name] = cvs[model_name].fit_transform(

        gen_text(klat, topic_df, model_params['intros_only'])

    )

    print('model_name={}'.format(model_name), flush=True)

    print('len(vocabulary_)={}'.format(len(cvs[model_name].vocabulary_)), flush=True)

    print('(num_docs, num_tokens)={}'.format(Xcvs[model_name].shape), flush=True)

    print('deleting {} stop words'.format(len(cvs[model_name].stop_words_)), flush=True)

    del cvs[model_name].stop_words_
tfidfs = {}

Xtfidfs = {}

for model_name, model_params in model_defs.items():

    print('model_name={}'.format(model_name))

    tfidfs[model_name] = TfidfTransformer()

    Xtfidfs[model_name] = tfidfs[model_name].fit_transform(Xcvs[model_name])
class ExplicitTopicModel:

    

    def __init__(self, cv, Xtfidf, topic_df):

        self.cv = cv

        self.Xtfidf = Xtfidf

        self.topic_df = topic_df

        self.tokenize = cv.build_analyzer()

        self.feature_names = np.array(cv.get_feature_names())

        print('creating explicit topic model (topics={}, tokens={})'.format(

            Xtfidf.shape[0], Xtfidf.shape[1]))

        

    def topn_topics_from_text(self, text, topn=10, thresh=0.0):

        tokens = self.tokenize(text)

        return self.topn_topics_from_tokens(tokens, topn=topn, thresh=thresh)



    def topn_topics_from_tokens(self, tokens, topn=10, thresh=0.0):

        topic_vector = self.topic_vec_from_tokens(tokens)

        return self.topn_topics_from_topic_vec(topic_vector, topn=topn, thresh=thresh)



    def topn_topics_from_topic_vec(self, topic_vector, topn=10, thresh=0.0):

        topic_indxs = np.argsort(-topic_vector)[:topn]

        top_topics_df = self.topic_df.iloc[topic_indxs].copy()

        topic_scores = topic_vector[topic_indxs]

        top_topics_df['score'] = topic_scores

        return top_topics_df[top_topics_df['score']>thresh].copy()

        

    def topic_vec_from_tokens(self, tokens):

        token_indices = [

            self.cv.vocabulary_[token] for token in tokens 

            if token in self.cv.vocabulary_]

        norm = max(1, len(token_indices))

        topic_vector = np.array(self.Xtfidf[:, token_indices].sum(axis=1)).squeeze() / norm

        return topic_vector

    

    def explain_topic_for_text(self, text, topic_title):

        text_tokens = self.tokenize(text)

        topic_tokens_df = self.topn_tokens_from_topic(topic_title, topn=1000)

        explanation = topic_tokens_df[topic_tokens_df['token'].isin(text_tokens)]

        return explanation.sort_values('score', ascending=False)

    

    def topn_tokens_from_topic(self, topic_title, topn=10):

        indx = self.topic_df.index[self.topic_df['title']==topic_title][0]

        token_vector = self.Xtfidf.getrow(indx).toarray().squeeze()

        token_indxs = np.argsort(-token_vector)[:topn]

        tokens = pd.DataFrame(

            zip(self.feature_names[token_indxs], token_vector[token_indxs]),

            columns=['token', 'score'])

        return tokens

        
etms = {}

for model_name, model_params in model_defs.items():

    etms[model_name] = ExplicitTopicModel(

        cvs[model_name], 

        Xtfidfs[model_name], 

        topic_dfs[model_params['topics']]

    )
def plot_etm_results(etms, model_defs, text):

    fig, axes = plt.subplots(4 , 2, figsize=(14, 17), sharex=True)

    results = pd.DataFrame()

    for ax, model_name in zip(axes.flatten(), model_defs):

        etm = etms[model_name]

        text_topics_df = etm.topn_topics_from_text(text, topn=10)

        text_topics_df['model'] = model_name

        results = pd.concat([results, text_topics_df])

        g = sns.barplot(

            x='score', y='title', color='orange', alpha=0.7, 

            data=text_topics_df, ax=ax).set_title(model_name)

    plt.tight_layout()

    return results
text1 = """

The canine - which was two months old when it died - has been

remarkably preserved in the permafrost of the Russian region, with its

fur, nose and teeth all intact.  DNA sequencing has been unable to determine

the species.  Scientists say that could mean the specimen represents an

evolutionary link between wolves and modern dogs."""
results = plot_etm_results(etms, model_defs, text1)
etm = etms['intros_small_12']

topics = etm.topn_topics_from_text(text1, topn=10)

for topic_title in topics['title']:

    explanation = etm.explain_topic_for_text(text1, topic_title)

    print(topic_title)

    print(explanation)

    print()
text2 = """

U.S. intelligence cannot say conclusively that Saddam Hussein

has weapons of mass destruction, an information gap that is complicating

White House efforts to build support for an attack on Saddam's Iraqi regime.

The CIA has advised top administration officials to assume that Iraq has

some weapons of mass destruction.  But the agency has not given President

Bush a "smoking gun," according to U.S. intelligence and administration

officials.

"""
results = plot_etm_results(etms, model_defs, text2)
etm = etms['intros_small_12']

topics = etm.topn_topics_from_text(text2, topn=10)

for topic_title in topics['title']:

    explanation = etm.explain_topic_for_text(text2, topic_title)

    print(topic_title)

    print(explanation)

    print()
text3 = """

The development of T-cell leukaemia following the otherwise

successful treatment of three patients with X-linked severe combined

immune deficiency (X-SCID) in gene-therapy trials using haematopoietic

stem cells has led to a re-evaluation of this approach.  Using a mouse

model for gene therapy of X-SCID, we find that the corrective therapeutic

gene IL2RG itself can act as a contributor to the genesis of T-cell

lymphomas, with one-third of animals being affected.  Gene-therapy trials

for X-SCID, which have been based on the assumption that IL2RG is minimally

oncogenic, may therefore pose some risk to patients.

"""
results = plot_etm_results(etms, model_defs, text3)
etm = etms['intros_small_12']

topics = etm.topn_topics_from_text(text3, topn=10)

for topic_title in topics['title']:

    explanation = etm.explain_topic_for_text(text3, topic_title)

    print(topic_title)

    print(explanation)

    print()
text4 = """

Share markets in the US plummeted on Wednesday, with losses accelerating 

after the World Health Organization declared the coronavirus outbreak a pandemic.

"""
results = plot_etm_results(etms, model_defs, text4)
etm = etms['intros_small_12']

topics = etm.topn_topics_from_text(text4, topn=10)

for topic_title in topics['title']:

    explanation = etm.explain_topic_for_text(text4, topic_title)

    print(topic_title)

    print(explanation)

    print()
cvs
Xtfidfs
topic_dfs
for key, model in cvs.items():

    file_name = "cv_{}.joblib".format(key)

    joblib.dump(model, file_name)
for key, df in topic_dfs.items():

    file_name = "topic_df_{}.csv".format(key)

    df.to_csv(file_name)
for key, mat in Xtfidfs.items():

    file_name = "xtfidf_{}.npz".format(key)

    scipy.sparse.save_npz(file_name, mat)
model_key = "intros_small"

topic_key = model_key.split('_')[1]

print('model_key={}'.format(model_key))

print('topic_key={}'.format(topic_key))
cv = joblib.load("cv_{}.joblib".format(model_key))

Xtfidf = scipy.sparse.load_npz("xtfidf_{}.npz".format(model_key))

topic_df = pd.read_csv("topic_df_{}.csv".format(topic_key), index_col="page_id")

etm = ExplicitTopicModel(cv, Xtfidf, topic_df)
etm.topn_topics_from_text(text3, topn=10)