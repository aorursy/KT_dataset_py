from datetime import datetime

deadline = datetime.strptime('2020-04-16 23:59:00','%Y-%m-%d %H:%M:%S')

print(deadline)

print(datetime.now())

print(deadline - datetime.now(), 'hours')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# imports

import plotly.express as px

import plotly.graph_objects as go

from scipy import stats

import numpy as np

import matplotlib.pyplot as plt

from scipy import interpolate

import json

import requests

import io



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import lognorm

from scipy.optimize import curve_fit

import string

from scipy.integrate import quad



from tqdm import tqdm



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.linear_model import SGDRegressor, LinearRegression, Lasso, Ridge, LogisticRegression

from sklearn.base import clone

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import LabelEncoder 



from wordcloud import WordCloud



# module settings

pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', 1000)

plt.rcParams['figure.figsize'] = [15, 8]



# https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf

# https://www.apsnet.org/edcenter/disimpactmngmnt/topc/EpidemiologyTemporal/Pages/ModellingProgress.aspx

meta = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

meta
# extracing title from json files

def parse_json(fn):

    with open(fn, 'r') as f:

        data = json.loads(f.read())

#     print(data)

    source = fn.split('/')[4]

#     print('source=',source)

    return([data['paper_id'], data['metadata']['title'], source, fn])



# parse_json('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/2c006d09b6fccc527bf5ee3de0f165b018c39e73.json')



def load_data():

    papers = []

    import os

    import json

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in tqdm(filenames):

            if '.json' not in filename:

                continue

            try:

                fn = os.path.join(dirname, filename)

#                 print(fn)

                papers.append(parse_json(fn))

            except Exception as e:

                print(e)

                continue

    return papers

            

papers = load_data()

papers


# to data frame

df = pd.DataFrame(papers,columns=['pmcid','title','source', 'file'])

df = df.drop_duplicates()

df = pd.merge(df, meta, on='pmcid', how='left').reset_index().rename(columns={'title_x':'title'})

del df['title_y']



# cleaning titles

titles = df['title']

titles_clean = []

for t in tqdm(titles):

    titles_clean.append(t.replace('"','').lower().strip())

#     for p in parts:



df['title_clean'] = titles_clean





# df = pd.read_csv('/kaggle/input/covid19-uncover-paper-titles/covid19_uncover_paper_titles.csv')

df['title'] = df['title'].astype('str')

df['title_clean'] = df['title_clean'].astype('str')



# text associated that we are interested in

inclusion = ('vulnerab epidem suscept copd male clinical illness female diabetes comorbid mortal prognosis hypertension blood heart liver lung kidney renal brain bladder tuberc drink alcohol mental psychi nerv smoking age stroke cardio cerebr cancer respira chronic factor risk population sex gender hospital health age population ethnic flu death mortality respir disease child pediat adult infant lung resp smok person patient old senior elderly infected individual').split(' ')



# text associated with more research

exclusion = ('pig rat mouse chicken horse mice dog monkey cat cations simulation bird gene cell tissue organism glia phagocy microbiology bacteria eukaryotic assay apoptosis signal protein pathway molecul rna dna monocytes chemokine').split(' ')



relevant_docs = set()

research_docs = set()

df['is_risk'] = 0

df['is_research'] = 0

for i in tqdm(df.index):

    t = df.loc[i,'title_clean']

    if np.nan == t or pd.isna(t):

        continue

    for w in inclusion:

#         print(f't={t}, w={w}')

        if w in t:

            df.loc[i, 'is_risk'] = 1

            relevant_docs.add(t)

            break

    for w in exclusion:

        if w in t:

            df.loc[i, 'is_research'] = 1

            research_docs.add(t)

            break





df.to_csv('paper_titles_is_risk.csv', index=False)

df.shape
df
df[['source','is_risk','is_research']].groupby('source').sum()
meta
# text associated that we are interested in

# inclusion = ('vulnerab suscept factor risk population sex gender hospital health age population ethnic flu death mortality respir disease child pediat adult infant lung resp smok person patient old senior elderly infected individual').split(' ')



# text associated with more research

# exclusion = ('pig', 'rat', 'mouse', 'mice', 'monkey', 'cations', 'gene', 'cell', 'tissue', 'glia', 'phagocy', 'microbiology', 'bacteria', 'eukaryotic', 'apoptosis', 'signal', 'protein', 'pathway', 'molecul', 'rna', 'dna', 'monocytes', 'chemokine')



# relevant_docs = set()

# research_docs = set()

# df['is_risk'] = 0

# df['is_research'] = 0

# for i in tqdm(range(df.shape[0])):

#     t = df.loc[i,'title_clean']

#     for w in inclusion:

#         if w in t:

#             df.loc[i, 'is_risk'] = 1

#             relevant_docs.add(t)

#     for w in exclusion:

#         if w in t:

#             df.loc[i, 'is_research'] = 1

#             research_docs.add(t)



# relevant_docs = list(relevant_docs)

# research_docs = list(research_docs)

# df.to_csv('paper_titles.csv', index=False)

df
df = df.query('is_risk == 1 & is_research == 0')

relevant_docs = df['title'].values



text = ' '.join(relevant_docs)

from wordcloud import WordCloud



# # Build word frequencies on filtered tokens

# freqs = pd.Series(np.concatenate([tokenize(x) for x in articles.Title])).value_counts()

# wordcloud(freqs, "Most frequent words in article titles tagged as COVID-19")



# Generate a word cloud image

wordcloud = WordCloud().generate(text)



# Display the generated image:

# the matplotlib way:

import matplotlib.pyplot as plt

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



# lower max_font_size

# wordcloud = WordCloud(max_font_size=40).generate(text)

# plt.figure()

# plt.imshow(wordcloud, interpolation="bilinear")

# plt.axis("off")

# plt.show()
# Fetch full text for 'risk' related papers  

def fetch_abstract(path):

    ret = []

    try:

        

        with open(path, 'r') as f:

            data = json.loads(f.read())

    #     print(data)

    #     for a in data['body_text']:



        for a in data['abstract']:

            ret.append(a['text'])

    except:

        pass

    return (' '.join(ret)).lower().strip()



# fetch_abstract('/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/26aec9a28a4345276498c14e302ead7d96c7feee.json')



# full_text = dict()

df['abstract'] = ''

for i in tqdm(df.index):

    fn = df.loc[i, 'file']

    df.loc[i, 'abstract'] = fetch_abstract(fn)

    

df.to_csv('paper_titles.csv', index=False)

df

    
abstract_text = ' '.join(df['abstract'].values)

abstract_text



# remove stop words

words = []

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 

stop_words = set(stopwords.words('english'))

abstract_text = ' '.join([w for w in word_tokenize(abstract_text) if w not in stop_words])



# Generate a word cloud image for the abstract

wordcloud = WordCloud().generate(abstract_text)



# Display the generated image:

# the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
pd.set_option('display.max_colwidth', -1)

print(df['title'].head(10))
df
from nltk.tokenize import sent_tokenize



results = []

LIMIT = 2000



prev_s = ''

for i in tqdm(df.index):

    a = df.loc[i,'abstract']

    for s in sent_tokenize(a):

#             if 'patient' in s and 'risk' in s:

        prev_s = s.strip()

        if ('covid' is s or 'corona' in s or 'sars' in s or 'mers' in s or 'h1n1' in s) and 'risk' in s:

            f = df.loc[i, 'file']

            title = df.loc[i, 'title']

            pid = df.loc[i, 'pmcid']

            results.append((pid, f,title, prev_s + ' ' + s.strip()))

#                 print(results)

#             print()

#             print('='*10 + fn + '='*10)

#             print(title)

#             print(s.strip())

            if len(results) > LIMIT:

                break

    if len(results) > LIMIT:

        break





with open('results_patient_risk.csv','w') as f:

    for pid,fn,title,s in results:

#         print()

#         print('='*10 + fn + '='*10)

#         print(title)

#         print(s.strip())

        f.write(fn + '\t' + title + '\t' + s.strip() + '\n')
# number of abstract sentences that matched grouped by paper titles

df_results = pd.DataFrame(results, columns=['pmcid', 'file','title','sentence'])

df_results.to_csv('results.csv')

print(df_results.shape)

df_results.groupby(['title']).count()[['pmcid']].sort_values(by='pmcid',ascending=False)
df_results[['pmcid', 'title','sentence']].head(1000)
results_text = ' '.join(df_results['sentence'].values)



# Generate a word cloud image for the abstract

wordcloud = WordCloud().generate(results_text)



# Display the generated image:

# the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

df_cases = pd.read_csv('/kaggle/input/covid19-european-enhanced-surveillance/covid19_enhanced_surveillance.csv')

df_cases.head(50)
pd.set_option('display.max_rows', 1000)

df_cases['cases'] = [ int(str(x).replace('<=5','5')) for x in df_cases['Cases'] ]

df_cases['deaths'] = [ int(str(x).replace('<=5','5')) for x in df_cases['Deaths'] ]

df_cases.groupby(['Outcome', 'Gender','Age group']).sum()