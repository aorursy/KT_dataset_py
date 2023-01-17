# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

import json

import glob

import sys



sys.path.insert(0, "../")



root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'



corona_features = {"doc_id": [None], "source": [None], "title": [None],

                  "abstract": [None], "text_body": [None]}

corona_df = pd.DataFrame.from_dict(corona_features)



json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)
def return_corona_df(json_filenames, df, source):



    for file_name in json_filenames:



        row = {"doc_id": None, "source": None, "title": None,

              "abstract": None, "text_body": None}



        with open(file_name) as json_data:

            data = json.load(json_data)



            doc_id = data['paper_id']

            row['doc_id'] = doc_id

            row['title'] = data['metadata']['title']



            # Now need all of abstract. Put it all in 

            # a list then use str.join() to split it

            # into paragraphs. 



            abstract_list = [abst['text'] for abst in data['abstract']]

            abstract = "\n ".join(abstract_list)



            row['abstract'] = abstract



            # And lastly the body of the text. 

            body_list = [bt['text'] for bt in data['body_text']]

            body = "\n ".join(body_list)

            

            row['text_body'] = body

            

            # Now just add to the dataframe. 

            

            if source == 'b':

                row['source'] = "BIORXIV"

            elif source == "c":

                row['source'] = "COMMON_USE_SUB"

            elif source == "n":

                row['source'] = "NON_COMMON_USE"

            elif source == "p":

                row['source'] = "PMC_CUSTOM_LICENSE"

            

            df = df.append(row, ignore_index=True)

    

    return df

    

corona_df = return_corona_df(json_filenames, corona_df, 'b')

corona_out = corona_df.to_csv('kaggle_covid-19_open_csv_format.csv')
corona_df = corona_df.dropna()
from IPython.display import FileLink

FileLink(r'kaggle_covid-19_open_csv_format.csv')
!pip install ktrain
import ktrain

ktrain.text.preprocessor.detect_lang = ktrain.text.textutils.detect_lang

df = corona_df

texts = df["text_body"]

tm = ktrain.text.get_topic_model(texts, n_topics=None, n_features=10000)
tm.print_topics()

tm.build(texts, threshold=0.25)
texts = tm.filter(texts)

df = tm.filter(df)
tm.visualize_documents(doc_topics=tm.get_doctopics())
transmission_results = tm.search('transmission', case_sensitive=False)

incubation_results = tm.search('incubation', case_sensitive=False)

environmental_results = tm.search('environmental stability', case_sensitive=False)
threshold = .80



transmission_topic_ids = {doc[3] for doc in transmission_results if doc[2]>threshold}

incubation_topic_ids = {doc[3] for doc in incubation_results if doc[2]>threshold}

environmental_topic_ids = {doc[3] for doc in environmental_results if doc[2]>threshold}



t_topics = transmission_topic_ids.copy()

t_topics.update(incubation_topic_ids)

t_topics.update(environmental_topic_ids)



tm.visualize_documents(doc_topics=tm.get_doctopics(t_topics))

docs = tm.get_docs(topic_ids=t_topics, rank=True)

print("TOTAL_NUM_OF_DOCS: %s" % len(docs))



print("##################################")



for t in t_topics:

    docs = tm.get_docs(topic_ids=[t], rank=True)

    print("NUM_OF_DOCS: %s" % len(docs))

    if(len(docs)==0): continue

    doc = docs[1]

    print('TOPIC_ID: %s' % (doc[3]))

    print('TOPIC: %s' % (tm.topics[t]))

    print('DOC_ID: %s'  % (doc[1]))

    print('TOPIC SCORE: %s '% (doc[2]))

    print('TEXT: %s' % (doc[0][0:400]))

    

    

    print("##################################")

tm.train_recommender()

text = "What is known about covid-19 transmission?"

for i, doc in enumerate(tm.recommend(text=text, n=5)):

    print('RESULT #%s'% (i+1))

    print('TEXT:\n\t%s' % (" ".join(doc[0].split()[:500])))

    print()
text = "What is known about covid-19 incubation period?"

for i, doc in enumerate(tm.recommend(text=text, n=5)):

    print('RESULT #%s'% (i+1))

    print('TEXT:\n\t%s' % (doc[0]))

    print()
text = "What is known about covid-19 environmental stability?"

for i, doc in enumerate(tm.recommend(text=text, n=5)):

    print('RESULT #%s'% (i+1))

    print('TEXT:\n\t%s' % (doc[0]))

    print()