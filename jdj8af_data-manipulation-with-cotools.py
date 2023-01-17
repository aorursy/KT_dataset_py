!pip install cord-19-tools==0.0.7
import cotools

from cotools import text, texts, abstract, abstracts

import spacy

import os

import sys





data = '/kaggle/input/CORD-19-research-challenge/2020-03-13'

os.listdir(data)
downloaded=True

if not downloaded:

    cotools.download("data")

#os.listdir('data')
comm_use = cotools.Paperset(f'{data}/comm_use_subset/comm_use_subset')

print(abstract(comm_use[0]))

print('-'*60)

print(text(comm_use[0]))
keys = comm_use.apply(lambda x: list(x.keys()))



print(set(sum(keys, [])))
covid_papers = [x for x in comm_use if any( c in text(x).lower() for c in ['covid', 'novel coronavirus'])]

# covid_papers = [x for x in comm_use if 'covid' in text(x).lower()]

len(covid_papers)
covid_text = [text(x) for x in covid_papers]

covid_abstracts = [abstract(x) for x in covid_papers]
covid_abstracts