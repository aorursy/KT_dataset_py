%matplotlib inline

from functools import reduce

import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from IPython.core.display import display, HTML
task_topics = {

    'theraputics': ['Effectiveness of drugs being developed and tried to treat COVID-19 patients',

                    'Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.',

                    'Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.',

                    'Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers'

                   ],

    'vaccines': ['Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.',

                 'Efforts targeted at a universal coronavirus vaccine.',

                 'Approaches to evaluate risk for enhanced disease after vaccination',

                 'Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]'

                 ],

    'animal_models' : ['Exploration of use of best animal models and their predictive value for a human vaccine.',

                        'Efforts to develop animal models and standardize challenge studies'

                      ]

    }
N_TITLES = 10  #Number of top titles to print for each task topic

FILTER_WORDS = ['sars','mers','influenza','respiratory','corona','cov'] #keyword strings used to filter titles




metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

print(f'meta data shape: {metadata.shape}')

has_title = metadata.title.apply(lambda x: str(x)!='nan')

metadata = metadata.iloc[has_title.values,:]

print(f'meta data shape after dropping docs without titles: {metadata.shape}')

metadata.head(n=2).transpose()
def has_word(title,filter_words):

    def has_this_word(has_one,word):

        if has_one:

            return True

        else:

            if word in title:

                return True

            else:

                return False

    return reduce(has_this_word,filter_words,False)



# filter to titles with relevant words



have_filter_word = metadata.title.apply(lambda x: has_word(x,FILTER_WORDS))  

metadata = metadata[have_filter_word]



# Convert metadata titles to TfIdf matrix

tfidf = TfidfVectorizer(stop_words='english',max_df = .75,min_df=2)

titles_mat = tfidf.fit_transform(metadata.title)

display(f'TfIdf matrix shape: {titles_mat.shape}')


for topic,topic_text in task_topics.items():

    display(HTML(f'Topic:  {topic}'))

    if isinstance(task_topics[topic], str): #in case of topic of only one str

        task_topics[topic] = np.array([task_topics[topic]])

    tasks_mat = tfidf.transform(task_topics[topic])

    task_rank = pd.DataFrame(np.mean(np.dot(titles_mat,np.transpose(tasks_mat)),axis=1),columns=['Rank'])

    top_tasks = task_rank.sort_values('Rank',ascending=False)[0:N_TITLES]

    #is_top_task = [True if x in top_tasks.index else False for x in metadata.index]

    df_s = metadata.iloc[top_tasks.index,:][['title','abstract','doi']]

    #convert to html

    df_s['title'] = '<span style="float: left; width: 100%; text-align: left;">' + df_s['title'] + '</span>'

    df_s['abstract'] = '<span style="float: left; width: 80%; text-align: left;">' + df_s['abstract'] + '</span>'

    df_s['doi'] = '<a href = "https://doi.org' + df_s['doi'] + '" target="_blank">link</a>'

    result = HTML(df_s.to_html(escape=False))

    display(result)