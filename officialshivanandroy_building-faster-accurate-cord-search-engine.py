# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import os

import json

import re

from tqdm import tqdm





dirs=["pmc_json","pdf_json"]

docs=[]

counts=0

for d in dirs:

    print(d)

    counts = 0

    for file in tqdm(os.listdir(f"../input/CORD-19-research-challenge/document_parses/{d}")):#What is an f string?

        file_path = f"../input/CORD-19-research-challenge/document_parses/{d}/{file}"

        j = json.load(open(file_path,"rb"))

        #Taking last 7 characters. it removes the 'PMC' appended to the beginning

        #also paperid in pdf_json are guids and hard to plot in the graphs hence the substring

        paper_id = j['paper_id']

        paper_id = paper_id[-7:]

        title = j['metadata']['title']



        try:#sometimes there are no abstracts

            abstract = j['abstract'][0]['text']

        except:

            abstract = ""

            

        full_text = ""

        bib_entries = []

        for text in j['body_text']:

            full_text += text['text']

                

        docs.append([paper_id, title, abstract, full_text])

        #comment this below block if you want to consider all files

        #comment block start

        counts = counts + 1

        if(counts >= 25000):

            break

        #comment block end    

df=pd.DataFrame(docs,columns=['paper_id','title','abstract','full_text']) 

print(df.shape)

df.head()
# installing haystack



! pip install git+https://github.com/deepset-ai/haystack.git
# importing necessary dependencies



from haystack import Finder

from haystack.indexing.cleaning import clean_wiki_text

from haystack.indexing.utils import convert_files_to_dicts, fetch_archive_from_http

from haystack.reader.farm import FARMReader

from haystack.reader.transformers import TransformersReader

from haystack.utils import print_answers
! wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz -q

! tar -xzf elasticsearch-7.6.2-linux-x86_64.tar.gz

! chown -R daemon:daemon elasticsearch-7.6.2



import os

from subprocess import Popen, PIPE, STDOUT

es_server = Popen(['elasticsearch-7.6.2/bin/elasticsearch'],

                   stdout=PIPE, stderr=STDOUT,

                   preexec_fn=lambda: os.setuid(1)  # as daemon

                  )

# wait until ES has started

! sleep 30
from haystack.database.elasticsearch import ElasticsearchDocumentStore

document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
# Now, let's write the dicts containing documents to our DB.

document_store.write_documents(df[['title', 'full_text']].rename(columns={'title':'name','full_text':'text'}).to_dict(orient='records'))
from haystack.retriever.sparse import ElasticsearchRetriever

retriever = ElasticsearchRetriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2-covid", use_gpu=True, context_window_size=1000)
finder = Finder(reader, retriever)
question = "What is the impact of coronavirus on babies?"

number_of_answers_to_fetch = 2



prediction = finder.get_answers(question=question, top_k_retriever=10, top_k_reader=number_of_answers_to_fetch)

print(f"Question: {prediction['question']}")

print("\n")

for i in range(number_of_answers_to_fetch):

    print(f"#{i+1}")

    print(f"Answer: {prediction['answers'][i]['answer']}")

    print(f"Research Paper: {prediction['answers'][i]['meta']['name']}")

    print(f"Context: {prediction['answers'][i]['context']}")

    print('\n\n')
question = "What is the impact of coronavirus on pregnant women?"

number_of_answers_to_fetch = 2



prediction = finder.get_answers(question=question, top_k_retriever=10, top_k_reader=number_of_answers_to_fetch)

print(f"Question: {prediction['question']}")

print("\n")

for i in range(number_of_answers_to_fetch):

    print(f"#{i+1}")

    print(f"Answer: {prediction['answers'][i]['answer']}")

    print(f"Research Paper: {prediction['answers'][i]['meta']['name']}")

    print(f"Context: {prediction['answers'][i]['context']}")

    print('\n\n')
question = "which organ does coronavirus impact?"

number_of_answers_to_fetch = 2



prediction = finder.get_answers(question=question, top_k_retriever=10, top_k_reader=number_of_answers_to_fetch)

print(f"Question: {prediction['question']}")

print("\n")

for i in range(number_of_answers_to_fetch):

    print(f"#{i+1}")

    print(f"Answer: {prediction['answers'][i]['answer']}")

    print(f"Research Paper: {prediction['answers'][i]['meta']['name']}")

    print(f"Context: {prediction['answers'][i]['context']}")

    print('\n\n')