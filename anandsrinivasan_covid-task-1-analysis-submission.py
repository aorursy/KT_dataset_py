# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

import json

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



def json_to_series(text):

    keys = []

    values= []

    if text:

        for key,value in text.items():

            keys.append(key)

            values.append(value)

    return pd.Series(values, index=keys)



def json_to_series_abtract(text):

    keys = []

    values= []

    if text:

        for key,value in text[0].items():

            keys.append(key)

            values.append(value)

    return pd.Series(values, index=keys)



dataset_final = pd.DataFrame()

for dirname, _, filenames in os.walk('../input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        with open(os.path.join(dirname, filename)) as json_data:

            data = json.load(json_data)

        dataset = pd.DataFrame.from_dict(data, orient='index').T.set_index('paper_id')

        dataset = pd.concat([dataset, dataset['metadata'].apply(json_to_series)], axis=1)

        dataset = pd.concat([dataset, dataset['abstract'].apply(json_to_series_abtract)], axis=1)

        if dataset_final.empty:

            dataset_final = dataset

        else:

            dataset_final = pd.concat([dataset_final,dataset], axis=0)



# Any results you write to the current directory are saved as output.
dataset_final = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')
dataset_final.columns
dataset_final['transmission_flag'] = dataset_final.abstract.str.contains('transmission',case=False,regex=False)
dataset_final.transmission_flag.sum()
dataset_final.shape
dataset_final[dataset_final.transmission_flag==1].abstract.tolist()[:10]
dataset_final['transmission_sentence'] = dataset_final.abstract.str.extract('([^.]*transmission[^.]*\.)',expand=False)

dataset_final['transmission_sentence'] = dataset_final['transmission_sentence'].replace(np.nan,None)
docs_transmission = list(set([x for x in dataset_final['transmission_sentence'].tolist() if x]))

len(docs_transmission)
docs_transmission[0:10]
import gensim

from gensim.summarization import summarize, summarize_corpus



combined_text = ".".join([str(x) for x in docs_transmission if len(str(x))>0])



summarize_texts = summarize(combined_text,ratio=0.05,split=True)



count = 1

for text in summarize_texts:

    print(str(count) + "." + text)

    count = count + 1
dataset_final['incubation_sentence'] = dataset_final.abstract.str.extract('([^.]*incubation[^.]*\.)',expand=False)

dataset_final['incubation_sentence'] = dataset_final['incubation_sentence'].replace(np.nan,None)

docs_incubation = list(set([x for x in dataset_final['incubation_sentence'].tolist() if x]))

len(docs_incubation)
docs_incubation[0:10]
combined_text = ".".join([str(x) for x in docs_incubation if len(str(x))>0])



summarize_texts = summarize(combined_text,ratio=0.1,split=True)



count = 1

for text in summarize_texts:

    print(str(count) + "." + text)

    count = count + 1