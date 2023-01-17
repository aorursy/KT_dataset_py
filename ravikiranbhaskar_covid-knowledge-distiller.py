#Install needed packages and NLP models

!pip install -U pysolr

!pip install -U scispacy

!pip install -U gensim

!pip install -U jsonpath-ng

!pip install -U pyvis

!pip install -U pyLDAvis

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz 

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_craft_md-0.2.4.tar.gz

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_jnlpba_md-0.2.4.tar.gz

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bionlp13cg_md-0.2.4.tar.gz    
!wget -O solr-8.5.0.zip "https://archive.apache.org/dist/lucene/solr/8.5.0/solr-8.5.0.zip";

!unzip solr-8.5.0.zip
!ls
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import os

import json

import math

import glob

import re

import time

import pysolr

import csv

import time

import scipy

import spacy

import scispacy

import multiprocessing

import pandas as pd

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

import ipywidgets as widgets

from os import path

from pandas import ExcelWriter

from pandas import ExcelFile

from jsonpath_ng.ext import parse

from collections import Counter

from collections import OrderedDict

from IPython.core.display import display, HTML

from IPython.display import IFrame

from pyvis.network import Network

from datetime import date

import dateutil.parser as dparser

import seaborn as sns

import matplotlib.pyplot as plt

import concurrent.futures

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True) #do not miss this line



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')





stopwords = nltk.corpus.stopwords.words('english')



custom_stop_words = [

    'doi', 'medRxiv','MedRxiv','preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 

    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI',

    '-PRON-', '-', 'medRxiv preprint'

]



stopwords.extend(custom_stop_words)
def clean_text(text) :

    text = re.sub(r" ?\([^)]*\)", "", text)

    text = re.sub(r" ?\[[^)]*\]", "", text)

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"  ", " ", text)

    return text.strip()
def preprocess_data() :

    print("Starting preprocessing of Data....\n","Please be patient, custom pre-processing can take approximately 25-30 mins....")

    start = time.time()



    meta_df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', sep=',', header=0)

    meta_columns = list(meta_df.columns)

    meta_df.fillna('', inplace=True)



    # Filter to pick only needed sections

    include_set = ['Abstract','Introduction', 'background', 'Discussion', 'Results', 'Results and Discussion', 'methods,results']



    # Throw execption is the Section Heads excel file is not available

    section_df = pd.DataFrame()

    try :

        print('Loading SECTIONHEADERS4PCKR...')

        section_df = pd.read_excel('/kaggle/input/sectionheaders4pckr/SECTIONHEADERS4PCKR.xlsx', sheet_name='Sheet1')

    except Exception as ex :

        print('Preprocessing cannot run without SECTIONHEADERS4PCKR.xlsx:', ex)

        raise

        

    section_df.dropna(subset=['Section Heads'], inplace=True)

    section_df = section_df[section_df['Section Heads'].isin(include_set)]

        

    # Categories/Targets to pick from Section Heads column

    introduction_categories = ["Introduction"]

    discussion_categories = ["Discussion"]

    result_categories = ["Results", "Results and Discussion", 'methods,results']



    introduction_df = section_df[section_df['Section Heads'].isin(introduction_categories)]

    discussion_df = section_df[section_df['Section Heads'].isin(discussion_categories)]

    result_df = section_df[section_df['Section Heads'].isin(result_categories)]



    intro_list = introduction_df.iloc[:, 0].tolist()

    discussion_list = discussion_df.iloc[:, 0].tolist()

    result_list = result_df.iloc[:, 0].tolist()



    intro_list = list(map(lambda x: str(x).strip(), intro_list))

    discussion_list = list(map(lambda x: str(x).strip(), discussion_list))

    result_list = list(map(lambda x: str(x).strip(), result_list))



    path = '/kaggle/input/CORD-19-research-challenge/'

    paths = [p for p in glob.glob(path + "**/*.json", recursive=True)]

    files_size = len(paths)



    col_names = ['paper_id','title','source', 'abstract','introduction','result','discussion','body', 'publish_time', 'has_covid','url']

    clean_df = pd.DataFrame(columns=col_names)



    covid_syns = ['COVID-19','COVID19','2019-nCoV','2019nCoV','Coronavirus','SARS-CoV-2','SARSCov2','novel Coronavirus']



    target_empty_count = 0



    abstract_expr = parse('$.abstract[*].text')



    for path in paths:

        with open(path) as f:

            intro_text_list = list()

            discussion_text_list = list()

            result_text_list = list()



            data = json.load(f)



            abstract_texts = [match.value for match in abstract_expr.find(data)]



            body_nodes = data['body_text']



            for entry in body_nodes :

                section_name = entry['section']

                section_name = section_name.strip().lower()

                entry_text = entry['text']



                if section_name.strip() in intro_list:

                    intro_text_list.append(entry_text)



                if section_name.strip() in discussion_list:

                    discussion_text_list.append(entry_text)



                if section_name.strip() in result_list:

                    result_text_list.append(entry_text)



            if len(intro_text_list) == 0 and len(discussion_text_list) == 0 and len(result_text_list) == 0 :

                target_empty_count = target_empty_count + 1





            id = data['paper_id']

            title = data['metadata']['title']

            

            url=''

            try :                 

                url = meta_df.loc[meta_df['sha'] == id, 'url'].iloc[0]

            except Exception as ex:

                pass            

            

            pubtime_df = meta_df[meta_df.sha == id]['publish_time']

            pubtime_dict = pubtime_df.to_dict()

            pubtime = ''

            for pubtime_field_key in pubtime_dict.keys():

                temp_pubtime_str = pubtime_dict.get(pubtime_field_key)

                orig_temp_pubtime_str = temp_pubtime_str

                try:

                    temppubdate = dparser.parse(orig_temp_pubtime_str,fuzzy=True).date()

                    pubtime = temppubdate.strftime("%Y-%m-%dT%H:%M:%SZ")

                except Exception as e:

                    temp_pubtime_str_parts = temp_pubtime_str.split(' ')

                    if len(temp_pubtime_str_parts) > 2 :

                        try :

                            temp_pubtime_str = temp_pubtime_str_parts[0] + ' ' + temp_pubtime_str_parts[1] + ' ' + temp_pubtime_str_parts[2]

                            temppubdate = dparser.parse(temp_pubtime_str,fuzzy=True).date()

                            pubtime = temppubdate.strftime("%Y-%m-%dT%H:%M:%SZ")

                        except Exception as ex:

                            pubtime = ''

                    else:

                        pubtime = ''             



            sha_df = meta_df[meta_df.sha == id]['source_x']

            meta_dict = sha_df.to_dict()

            source = ''

            for meta_field_key in meta_dict.keys():

                source = meta_dict.get(meta_field_key)



            if not source:

                title_df = meta_df[meta_df.title == title]['source_x']

                meta_dict = title_df.to_dict()

                for meta_field_key in meta_dict.keys():

                    source = meta_dict.get(meta_field_key)



            abstract = clean_text(" ".join(abstract_texts))

            introduction = clean_text(" ".join(intro_text_list))

            discussion = clean_text(" ".join(discussion_text_list))

            result = clean_text(" ".join(result_text_list))

            body = " ".join([introduction, discussion, result])



            has_covid = 'false'



            res = [ele for ele in covid_syns if (ele.lower() in body.lower())]

            if(len(res)  > 0):

                has_covid = 'true'



            if len(body.strip()) > 0 or len(abstract) > 0:

                new_row = {'paper_id': id, 'title': title.strip(), 'source': source,'abstract': abstract.strip(),

                           'introduction': introduction.strip(),'result': result.strip(),'discussion': discussion.strip(),

                            'body': body.strip(), 'publish_time': pubtime,'has_covid': has_covid, 'url':url}

                clean_df = clean_df.append(new_row, ignore_index=True)



    # Drop duoplicate papers

    clean_df.drop_duplicates(subset=['title','abstract'], keep='first', inplace=False)

    clean_df.to_csv('/kaggle/working/CORD-19.csv', index=True)



    print('Final DataFrame Shape - ', clean_df.shape)

    print("Papers that dont have Intro, Discussion or Result  - ", target_empty_count)

    print('Total Papers processed - ', files_size)



    print('Time Elaspsed - ', time.time() - start)

    
!solr-8.5.0/bin/solr start -force
!solr-8.5.0/bin/solr create -c covid19 -s 1 -rf 1 -force
# Using _default configset with data driven schema functionality. NOT RECOMMENDED for production use.

!solr-8.5.0/bin/solr config -c covid19 -p 8983 -action set-user-property -property update.autoCreateFields -value false
#Set Up Synonyms



!echo 'COVID-19,covid19,2019-nCoV,2019nCoV,Coronavirus,SARS-CoV-2,SARSCov2,novel Coronavirus' > solr-8.5.0/server/solr/covid19/conf/synonyms.txt;

!echo 'heart,cardiac,tachycardia,myocardial' >> solr-8.5.0/server/solr/covid19/conf/synonyms.txt;

!echo 'pulmonary,respiratory' >> solr-8.5.0/server/solr/covid19/conf/synonyms.txt;
!cat solr-8.5.0/server/solr/covid19/conf/synonyms.txt
#Reload the covid19 core/collection because we added new synonyms. Need reload as it will affect index

#Whenever new synonyms are added we need to reindex as synonyms are applied both on index and query analyzers

!curl 'http://localhost:8983/solr/admin/cores?action=RELOAD&core=covid19'
#Add custom field Type that wont tokenize phrases for fields like source etc

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field-type" : {"name":"keywordText","class":"solr.TextField", "positionIncrementGap":"100", "indexAnalyzer" : {"tokenizer":{"class":"solr.KeywordTokenizerFactory" }, "filters":[{"class":"solr.TrimFilterFactory"},{"class":"solr.StopFilterFactory", "ignoreCase":true, "words":"lang/stopwords_en.txt"},{"class":"solr.ManagedSynonymGraphFilterFactory", "managed":"english" },{"class":"solr.RemoveDuplicatesTokenFilterFactory"},{"class":"solr.FlattenGraphFilterFactory"}]},"queryAnalyzer" : {"tokenizer":{"class":"solr.KeywordTokenizerFactory" },"filters":[{"class":"solr.TrimFilterFactory"},{"class":"solr.StopFilterFactory", "ignoreCase":true, "words":"lang/stopwords_en.txt"},{"class":"solr.ManagedSynonymGraphFilterFactory", "managed":"english" },{"class":"solr.RemoveDuplicatesTokenFilterFactory"}]}}}' http://localhost:8983/solr/covid19/schema
#Create SOLR field definitions

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"title", "type":"text_en_splitting_tight", "multiValued":false, "stored":true, "indexed":true}}' http://localhost:8983/solr/covid19/schema;

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"abstract", "type":"text_en_splitting_tight", "multiValued":false, "stored":true, "indexed":true}}' http://localhost:8983/solr/covid19/schema;

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"source", "type":"keywordText", "multiValued":false, "stored":true, "indexed":true}}' http://localhost:8983/solr/covid19/schema;

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"introduction", "type":"text_en_splitting_tight", "multiValued":false, "stored":true, "indexed":true}}' http://localhost:8983/solr/covid19/schema;

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"discussion", "type":"text_en_splitting_tight", "multiValued":false, "stored":true, "indexed":true}}' http://localhost:8983/solr/covid19/schema;

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"result", "type":"text_en_splitting_tight", "multiValued":false, "stored":true, "indexed":true}}' http://localhost:8983/solr/covid19/schema;

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"body", "type":"text_en_splitting_tight", "multiValued":false, "stored":true, "indexed":true}}' http://localhost:8983/solr/covid19/schema;

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"publish_time", "type":"pdate", "multiValued":false, "stored":true, "indexed":true}}' http://localhost:8983/solr/covid19/schema;

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"url", "type":"string", "multiValued":false, "stored":true, "indexed":true}}' http://localhost:8983/solr/covid19/schema;        

!curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"has_covid", "type":"boolean", "multiValued":false, "stored":true, "indexed":true}}' http://localhost:8983/solr/covid19/schema;
solr = pysolr.Solr('http://localhost:8983/solr/covid19/', timeout=10)
generic_model = spacy.load('en_core_sci_md')



#Load preprocessed CSV data

public_csv_path = '/kaggle/input/cord19s4pckr/CORD19S4PCKR.csv'

csv_path = '/kaggle/working/CORD19S4PCKR.csv'



if not path.exists(public_csv_path):

    print('Calling Preprocessing...')

    preprocess_data()

else :

    csv_path = public_csv_path

    print('Dataset Path - ', csv_path)

    

df = pd.read_csv(csv_path, sep=',', header=0)

#df.dropna(axis=0, how='all', thresh=None, subset=['title','abstract','body'], inplace=False)

df.fillna('', inplace=True)

df.drop_duplicates(subset=['title','abstract'], keep='first', inplace=True)

df = df[df[['abstract','body']].ne('').any(axis=1)]

print('DF candidate_list size - ', df.shape)



df.head(2)
# Index each pandas row as a document into SOLR search engine



covid_syns = ('SARSCoV2','SARS-CoV-2', '2019-nCoV','2019nCoV','COVID-19', 'COVID19','coronavirus', 'corona virus' 'novel coronavirus')



list_for_solr=[]

counter = 0

for index, row in df.iterrows():

    id = row['paper_id']

    title = row["title"]

    source = row["source"]    

    abstract = row["abstract"]

    introduction = row["introduction"]

    discussion = row["discussion"]

    result = row["result"]

    publish_time = row["publish_time"]

    body = row["body"]  # Cocatenated text of all text fields abstract, introduction, discussion, result

    

    if((title and title.isspace()) and (abstract and abstract.isspace()) and (body and body.isspace())):

        continue

        

    has_covid = 'false'

    if any(words in body for words in covid_syns):

        has_covid = 'true'

    

    solr_content = {}

    solr_content['id'] = id

    solr_content['title'] = title

    solr_content['source'] = source

    solr_content['abstract'] = abstract

    solr_content['introduction'] = introduction

    solr_content['discussion'] = discussion

    solr_content['result'] = result    

    solr_content['body'] = body  

    solr_content['has_covid'] = has_covid

    

    if publish_time != '':

        solr_content['publish_time'] = publish_time    

        

    list_for_solr.append(solr_content)

    

    if index % 1000 == 0:

        solr.add(list_for_solr)

        list_for_solr = []

        counter = counter + 1000

        print('Indexed Papers - ', counter)

        

#Commit is very costly use it sparingly        

solr.commit()

print('Indexing Finished !')
def extract_entities(scimodels, text) :

    entities = {}

    

    for nlp in scimodels :

        doc = nlp(text)

        for ent in doc.ents:

            entity = ent.text

            if ent.label_ in entities :

                if entities[ent.label_].count(ent.text) == 0:

                    entities[ent.label_].append(ent.text)

            else :

                entities[ent.label_] = [ent.text]



    return entities
def initilize_nlp_models(model_names):

    scimodels = {}

    for name in model_names:

        scimodels[name] = spacy.load(name)

    

    print('Models Loaded')

    return scimodels
def search_task_answers(search_results, display_on=False) :

    answers_list = list()

    

    for search_result in search_results:

        doc_hl_dict = {}

        

        id = search_result.get('id', "MISSING")

        title = search_result.get('title', "MISSING")

        source = search_result.get('source', "MISSING")        

        publish_time = search_result.get('publish_time')

        url = search_result.get('url', 'MISSING')

        

        doc_highlights = search_results.highlighting[id]

        

        doc_hl_dict['id'] = id

        doc_hl_dict['title'] = title

        doc_hl_dict['source'] = source

        doc_hl_dict['publish_time'] = publish_time

        doc_hl_dict['url'] = url        

                

        if len(doc_highlights) > 0 and display_on:

            display(HTML(f'<h4><i>{title}\n</i></h4>'))



        for doc_hl_field in doc_highlights:

            hl_snippets = doc_highlights[doc_hl_field]

        

            if len(hl_snippets) > 0 :

                answer_snippet = ''

                

                if display_on :

                    display(HTML(f'\t<h5>{doc_hl_field}\n</h5>\n'))

            

                for index, snippet in enumerate(hl_snippets, start=1):

                    answer_snippet = answer_snippet.strip() + " " + snippet.strip()

                    

                    if display_on :

                        display(HTML(f'<blockquote>{index}. {snippet.strip()}\n</blockquote>'))

                                  

                doc_hl_dict[doc_hl_field] = answer_snippet.strip()

                    

        if len(doc_hl_dict) > 0:

            answers_list.append(doc_hl_dict)

        

    return answers_list
def search(query, rows=5, mark=False):

    # Search for data

    if mark :

        search_results = solr.search(query, **{

        'fq':'has_covid:true',

        'rows' : rows,

        'qf':'title^50.0 abstract^40.0 introduction^30.0 discussion^20.0 result^50.0 body^10.0',

        'pf':'title^60.0 abstract^50.0 introduction^40.0 discussion^30.0 result^60.0 body^20.0',

        'hl': 'true',

        'hl.bs.type': 'SENTENCE',

        'hl.method' : 'unified',

        'hl.snippets' : 5,

        'hl.usePhraseHighlighter': 'true',

        'hl.highlightMultiTerm' : 'true',

        'hl.tag.pre':'<mark>',

        'hl.tag.post':'</mark>',

        'df':'body',

        'hl.fl':'introduction,discussion,result'

        })

    else:

        search_results = solr.search(query, **{

        'fq':'has_covid:true',

        'rows' : rows,

        'qf':'title^50.0 abstract^40.0 introduction^30.0 discussion^20.0 result^50.0 body^10.0',

        'pf':'title^60.0 abstract^50.0 introduction^40.0 discussion^30.0 result^60.0 body^20.0',

        'hl': 'true',

        'hl.bs.type': 'SENTENCE',

        'hl.method' : 'unified',

        'hl.snippets' : 5,

        'hl.usePhraseHighlighter': 'true',

        'hl.highlightMultiTerm' : 'true',

        'hl.tag.pre':'',

        'hl.tag.post':'',

        'df':'body',

        'hl.fl':'introduction,discussion,result'

        })        



    num_docs_found = search_results.hits

    num_search_results = len(search_results)

    

    return num_docs_found, search_results
# Map entities existence in intro-result-discussion respectively to label values

label_def = {'111':'prior-newdata','101':'prior-strong','100':'prior','001':'speculative','010':'unknown', '011':'novel'}



model_names = ['en_ner_craft_md','en_ner_jnlpba_md','en_ner_bc5cdr_md','en_ner_bionlp13cg_md']

scimodels = initilize_nlp_models(model_names)
def populate_labels(task_answers) :

    field_list = ['introduction','discussion', 'result']



    for doc_answer_dict in task_answers:

        all_entities = set()

        intro_entities = set()

        discussion_entities = set()

        result_entities = set() 



        for field_name, answer_text in doc_answer_dict.items():

            if field_name in field_list:

                chosen_models = list(scimodels.values())

                

                ent_dict = extract_entities(chosen_models, answer_text)

                ent_list = set()

                

                for ner_type, model_ent_list in ent_dict.items():

                    ent_list.update(model_ent_list)                



                all_entities.update(ent_list)

                if field_name == 'introduction' :

                    intro_entities.update(ent_list)

                elif field_name == 'discussion' :

                    discussion_entities.update(ent_list)

                else :

                    result_entities.update(ent_list)



        # Now set up labels for entities

        prior_newdata_entities = set()

        prior_strong_entities = set()

        prior_entities = set()

        speculative_entities = set()

        unknown_entities = set()    

        novel_entities = set()



        for a_ent in all_entities :

            if a_ent in intro_entities and a_ent in result_entities and a_ent in discussion_entities : 

                prior_newdata_entities.add(a_ent)

            elif a_ent in intro_entities and a_ent not in result_entities and a_ent in discussion_entities :

                prior_strong_entities.add(a_ent)

            elif a_ent in intro_entities and a_ent not in result_entities and a_ent not in discussion_entities :

                prior_entities.add(a_ent)

            elif a_ent not in intro_entities and a_ent in result_entities and a_ent not in discussion_entities : 

                unknown_entities.add(a_ent)

            elif a_ent not in intro_entities and a_ent in result_entities and a_ent in discussion_entities :

                novel_entities.add(a_ent)

            else :

                pass

            

        if(len(prior_newdata_entities) > 0) :

            doc_answer_dict['prior-newdata'] = list(prior_newdata_entities)



        if(len(prior_strong_entities) > 0) :

            doc_answer_dict['prior-strong'] = list(prior_strong_entities)          



        if(len(prior_entities) > 0) :

            doc_answer_dict['prior'] = list(prior_entities)             



        if(len(speculative_entities) > 0) :

            doc_answer_dict['speculative'] = list(speculative_entities)



        if(len(unknown_entities) > 0) :

            doc_answer_dict['unknown'] = list(unknown_entities)           



        if(len(novel_entities) > 0) :

            doc_answer_dict['novel'] = list(novel_entities)           



    return task_answers
tasks = ['Smoking and pre-existing pulmonary disease', 

         'Co-infections, co-morbidities and respiratory infections',

         'Neonates and pregnant women',

         'Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.',

         'Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors',

         'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',

         'Susceptibility of populations',

         'Public health mitigation measures that could be effective for control'

        ]



queries = ['("Smoking COVID-19"~10 OR "tobacco COVID-19"~10 OR "nicotine COVID-19"~10 OR "pulmonary disease"~10)', 

           '("Co-infections COVID-19"~10 OR "co-morbidities COVID-19"~10 OR "respiratory infections COVID-19"~10)',

           '("Neonates COVID-19"~10 OR "pregnant women COVID-19"~10)',

           '(Socio-economic OR "behavioral factors"~10 OR "economic impact"~10)',

           '("Transmission dynamics"~10 OR "reproductive number"~3 OR "incubation period"~3 OR "serial interval"~10 OR "modes of transmission"~10 OR "environmental factors"~10)',

           '("Severity of disease"~10 OR "risk of fatality"~10 OR "symptomatic patients"~10 OR "high-risk patient"~10)',

           'Susceptibility of populations',

           '("mitigation measures"~10 OR "effective control"~10)'

        ]



# Map entities existence in intro-result-discussion respectively to label values

label_def = {'111':'prior-newdata','101':'prior-strong','100':'prior','001':'speculative','010':'unknown', '011':'novel'}
# Label based Bubble chart over time

label_def_list = list(label_def.values())

label_weights = {"prior-newdata":6, "prio-strong":5, "prior":4, "novel":3, "speculative":2, "unknown":1}

label_weight_values = list(label_weights.values())

bcol_names = ['title','source', 'entity', 'label','weight','publish_time']



for task,query in zip(tasks, queries):

    

    label_pca_df = pd.DataFrame()

    label_df = pd.DataFrame(columns=bcol_names)

    

    label_entities = []

    num_docs_found, search_results = search(query, 300, False)

    num_search_results = len(search_results)

    

    all_entities_set = set()

    

    if num_docs_found > 0:

        task_answers = search_task_answers(search_results, False)

        task_answers = populate_labels(task_answers)

        

        for task_answer in task_answers :

            title = task_answer['title']

            source = task_answer['source']

            publish_time = task_answer['publish_time']

            

            present_labels = set(list(task_answer.keys()))&set( list(label_weights.keys()))

                        

            for present_label in present_labels :

                label_entities = task_answer[present_label]

                

                all_entities_set.update(label_entities)

                

                for ent in label_entities :

                    ldict = {'title':title,'source':source, 'entity':ent, 

                             'label':present_label, 'weight':label_weights[present_label], 

                             'publish_time':publish_time}

                    label_df = label_df.append(ldict, ignore_index=True)

        

        

        label_pca_df = pd.DataFrame(0, index=np.arange(len(task_answers)), columns=all_entities_set)

        title_list = []

        for l_index, task_answer in enumerate(task_answers) :

            title_list.append(task_answer['title'])

            

            present_labels = set(list(task_answer.keys()))&set( list(label_weights.keys()))

            for present_label in present_labels :

                label_entities = task_answer[present_label]            

                for ent in label_entities :

                    label_pca_df.set_value(l_index, ent, label_weights[present_label])

        



        # PCA on entities with Labels like prior, novel etc#

        ###################################################

        l_pca = PCA(n_components=2)

        l_data = l_pca.fit_transform(label_pca_df)

        

        # Scree Variance Plot

        l_var_df = pd.DataFrame({"variance": l_pca.explained_variance_ratio_, "PC":["PC1", "PC2"]})       

        

        lvar_fig = px.bar(l_var_df, x='PC', y='variance', 

                          labels={'variance':'Variance Explained', 'PC':'Principal Components'} ,

                          title="Scree Plot - " + task)

        py.offline.iplot(lvar_fig)

        

#         #Elbow

#         sum_squared_distances = []

#         K = range(1,12)

#         for k in K:

#             km = KMeans(n_clusters=k, init = 'k-means++', random_state = 13)

#             km = km.fit(l_data)

#             print(km.inertia_)

#             sum_squared_distances.append(km.inertia_)  



#         plt.plot(K, sum_squared_distances, 'bx-')

#         plt.xlabel('k')

#         plt.ylabel('Sum_of_squared_distances')

#         plt.title('Elbow Method For Optimal k')

#         plt.show()        

        

        # KMeans Cluster Plot. The Elbow method showed k = 3 as best

        l_kmeans_indices = KMeans(n_clusters=3, init = 'k-means++', random_state = 13).fit_predict(l_data)

        colors = ['b', 'g', 'r', 'c', 'y', 'm']

        l_pca_dict = {'PC1': l_data[:,0], 'PC2': l_data[:,1], 'Title':title_list, 'Color':[colors[d] for d in l_kmeans_indices]}

        l_pca_df = pd.DataFrame(l_pca_dict)        

        l_fig = px.scatter(l_pca_df, x="PC1", y="PC2", color='Color', hover_name="Title", title=task)        

        py.offline.iplot(l_fig)

    

    

    # Bubble Plot

    label_df["weight"].fillna(1, inplace = True)

    label_df["publish_time"]= pd.to_datetime(label_df["publish_time"])



    bubble_time = px.scatter(label_df.sort_values(by='publish_time',ascending=True),

                        x="publish_time", y="entity", color="label", 

                        size="weight",hover_name="title",

                        title=task)

    bubble_time.update_yaxes(showticklabels=False)

    py.offline.iplot(bubble_time)    

    

    bubble_entity = px.scatter(label_df.sort_values(by='publish_time',ascending=True),

                        x="entity", y="label", color="source", 

                        size="weight",hover_name="title",

                        title=task) 

    bubble_entity.update_xaxes(showticklabels=False)

    py.offline.iplot(bubble_entity)

    
label_def_list = list(label_def.values())



for task,query in zip(tasks, queries):

    label_entities = []

    numDocsFound, search_results = search(query,10, True)

    num_search_results = len(search_results)

    

    display(HTML(f'<h3 style="color:red">Task - {task} \n</h3>'))

    display(HTML(f'<h3 style="color:blue">Top {num_search_results} search result(s) of {numDocsFound} total. \n</h3>'))

    if numDocsFound > 0:

        task_answers = search_task_answers(search_results, True)

        task_answers = populate_labels(task_answers)
# PCA via DOC-TERM MATRIX 

for task,query in zip(tasks, queries):



    num_docs_found, search_results = search(query, 100, False)

    uniq_entity_list = set()

    

    q_entity_list = []

    title_list = []

    body_list = []

    

    for index,search_result in enumerate(search_results):

        title = search_result.get('title', "")        

        title_list.append(title.strip())

        

        body = search_result.get('body', "")

        body_list.append(body)

                

        doc_ent_list = []

        

        for scimodel in scimodels.values(): 

            body_doc = scimodel(body)

            doc_ent_list = doc_ent_list + [e.text.lower() for e in body_doc.ents]

            

        uniq_entity_list.update(doc_ent_list)

        q_entity_list.extend(doc_ent_list)

    

    doc_df = pd.DataFrame(data=body_list, columns=['documents'])

    

    vectorizer = CountVectorizer(vocabulary=uniq_entity_list, analyzer='word', min_df=2, 

                                 max_features = 5000, preprocessor=None, tokenizer=None, 

                                 lowercase=True, stop_words=stopwords)

    

    vectors = vectorizer.fit_transform(doc_df['documents'].values)

    

    title_df = pd.DataFrame(data=title_list, columns=['title'])

    

    matrix_df = pd.DataFrame(data=vectors.toarray(), columns=vectorizer.get_feature_names())

    

    pca = PCA(n_components=3)

    pca_components = pca.fit_transform(vectors.toarray())

    

    # Scree Variance Plot

    var_df = pd.DataFrame({"variance": pca.explained_variance_ratio_, "PC":["PC1", "PC2", "PC3"]})       

        

    var_fig = px.bar(var_df, x='PC', y='variance', 

                      labels={'variance':'Variance Explained', 'PC':'Principal Components'} ,

                      title="Scree Plot - " + task)

    py.offline.iplot(var_fig)    



    x_axis = [o[0] for o in pca_components]

    y_axis = [o[1] for o in pca_components]

    z_axis = [o[2] for o in pca_components] 

    

    colors = ["r", "b", "c", "y", "m", ]

    

    #The Elbow method showed k = 3 as best

    kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 13)

    kmean_indices = kmeans.fit_predict(pca_components)    

    

    pca_dict = {'PC1': x_axis, 'PC2': y_axis, 'PC3': z_axis,'Title':title_list, 'Color':[colors[d] for d in kmean_indices]}

    pca_df = pd.DataFrame(pca_dict)

    fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z='PC3',color='Color', hover_name="Title", title=task)

    py.offline.iplot(fig)

    

    #Bar Graphs of Entity Frequencies

    nerCntr = Counter(q_entity_list)

    freq_ners = nerCntr.most_common(50)



    x,y = zip(*freq_ners)

    x,y = list(x),list(y)



    plt.figure(figsize=(15,10))

    ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

    plt.xlabel('Entity')

    plt.xticks(rotation=90)

    plt.ylabel('Frequency')

    plt.title(task)

    plt.show()    

    
def search_custom_query(query, num_docs=10, mark=False, display_on=False):

    numDocsFound, search_results = search(query, num_docs, mark)

    if numDocsFound > 0:

        task_answers = search_task_answers(search_results, display_on)

        task_answers = populate_labels(task_answers)

        

        return search_results,task_answers
search_terms = 'Pregnant women'

searchbar = widgets.interactive(search_custom_query, query=search_terms, num_docs=5,  mark=False, display_on=False)

searchbar
temp_query = searchbar.kwargs

searched_query = temp_query['query']



if searched_query:

    s,a = search_custom_query(searched_query, num_docs=50, mark=False, display_on=False)

else:

    s,a = search_custom_query('("Smoking COVID-19"~10 OR "Nicotine COVID-19"~10 OR "tobacco COVID-19"~10 OR "pulmonary disease"~10)', num_docs=200, mark=False, display_on=False)



search_results = s

task_answers = a

OR
def cosine_score(x):

        d = []

        for i in range(len(x)):

            for j in range(i+1,len(x)):

                doc1= generic_model(x[i])

                doc2= generic_model(x[j])

                d.append({



                    'Title1': x[i],

                    'Title2': x[j],

                    'Score': doc1.similarity(doc2)*30

                }

            )



        return d



class network_graph:

    

    def __init__(self,search_result):

        data = pd.DataFrame(search_result)

        self.data = data

        col_list = data.columns

        drop_list = ['introduction','discussion','result']

        self.new_cols = []

        for cols in col_list:

            if cols not in drop_list:

                self.new_cols.append(cols)



        

    def extract_titles(self,search_result):

        title_list = []

        for title in range(len(search_result)):

            title_list.append(search_result[title]['title'])

            

        title_dict_temp = cosine_score(title_list)

        self.title_df = pd.DataFrame(title_dict_temp)

        title_df_temp1 = self.title_df[['Title1']]

        title_df_temp1.rename(columns={'Title1':'Title'},inplace=True)

        title_df_temp2 = self.title_df[['Title2']]

        title_df_temp2.rename(columns={'Title2':'Title'},inplace=True)

        self.merged_titles = pd.concat([title_df_temp1,title_df_temp2],axis=0)

        

    

    def extract_words(self,search_result):

        

        word_df = self.data[self.new_cols]

        

        word_df_temp = word_df.drop(['id','title'],axis=1)

        

        col_list = list(word_df_temp.columns)

        

        self.word_df_prior =pd.DataFrame()

        self.word_df_prior_strong =pd.DataFrame()

        self.word_df_prior_newdata =pd.DataFrame()

        self.word_df_speculative =pd.DataFrame()

        self.word_df_unknown =pd.DataFrame()

        self.word_df_novel =pd.DataFrame()

        

        word_df_prior_1 =pd.DataFrame()

        word_df_prior_strong_1 =pd.DataFrame()

        word_df_prior_newdata_1 =pd.DataFrame()

        word_df_speculative_1 =pd.DataFrame()

        word_df_unknown_1 =pd.DataFrame()

        word_df_novel_1 =pd.DataFrame()

            

        for col in col_list:

            if col =='prior':

                self.word_df_prior = word_df[['id','title','prior']]

                self.word_df_prior = self.word_df_prior.explode('prior')

                self.word_df_prior.dropna(subset = ["prior"], inplace=True)

                self.word_df_prior['Weight'] = 8

                

                word_df_prior_1 = self.word_df_prior[['prior']]

                word_df_prior_1.drop_duplicates(inplace=True)

                word_df_prior_1['Color'] = 'tomato'

                word_df_prior_1['Size'] = 10

                word_df_prior_1.columns = ['Words','Color','Size']

                

            elif col== 'prior-strong':

                self.word_df_prior_strong = word_df[['id','title','prior-strong']]

                self.word_df_prior_strong = self.word_df_prior_strong.explode('prior-strong')

                self.word_df_prior_strong.dropna(subset = ["prior-strong"], inplace=True)

                self.word_df_prior_strong['Weight'] = 6

                

                word_df_prior_strong_1 = self.word_df_prior_strong[['prior-strong']]

                word_df_prior_strong_1.drop_duplicates(inplace=True)

                word_df_prior_strong_1['Color'] = 'sienna'

                word_df_prior_strong_1['Size'] = 10

                word_df_prior_strong_1.columns = ['Words','Color','Size']

                

                

            elif col == 'prior_newdata':

                self.word_df_prior_newdata = word_df[['id','title','prior_newdata']]

                self.word_df_prior_newdata = self.word_df_prior_newdata.explode('prior_newdata')

                self.word_df_prior_newdata.dropna(subset = ["prior_newdata"], inplace=True)

                self.word_df_prior_newdata['Weight'] = 4

                

                word_df_prior_newdata_1 = self.word_df_prior_newdata[['prior_newdata']]

                word_df_prior_newdata_1.drop_duplicates(inplace=True)

                word_df_prior_newdata_1['Color'] = 'bisque'

                word_df_prior_newdata_1['Size'] = 10

                word_df_prior_newdata_1.columns = ['Words','Color','Size']

                

                

            elif col =='speculative':

                self.word_df_speculative = word_df[['id','title','speculative']]

                self.word_df_speculative = self.word_df_speculative.explode('speculative')

                word_df_speculative.dropna(subset = ["speculative"], inplace=True)

                self.word_df_speculative['Weight'] = 10

                

                word_df_speculative_1 = self.word_df_speculative[['speculative']]

                word_df_speculative_1.drop_duplicates(inplace=True)

                word_df_speculative_1['Color'] = 'dimgray'

                word_df_speculative_1['Size'] = 10

                word_df_speculative_1.columns = ['Words','Color','Size']

                

            elif col =='unknown':

                self.word_df_unknown = word_df[['id','title','unknown']]

                self.word_df_unknown = self.word_df_unknown.explode('unknown')

                self.word_df_unknown.dropna(subset = ["unknown"], inplace=True)

                self.word_df_unknown['Weight'] = 30

                

                word_df_unknown_1 = self.word_df_unknown[['unknown']]

                word_df_unknown_1.drop_duplicates(inplace=True)

                word_df_unknown_1['Color'] = 'black'

                word_df_unknown_1['Size'] = 10

                word_df_unknown_1.columns = ['Words','Color','Size']

                

            elif col =='novel':

                self.word_df_novel = word_df[['id','title','novel']]

                self.word_df_novel = self.word_df_novel.explode('novel')

                self.word_df_novel.dropna(subset = ["novel"], inplace=True)

                self.word_df_novel['Weight'] = 2

                

                word_df_novel_1 = self.word_df_novel[['novel']]

                word_df_novel_1.drop_duplicates(inplace=True)

                word_df_novel_1['Color'] = 'purple'

                word_df_novel_1['Size'] = 10

                word_df_novel_1.columns = ['Words','Color','Size']

                

    # Take all the titles and categorize them into one group

    

        title_df_1 = self.merged_titles[['Title']]

        title_df_1.drop_duplicates(inplace=True,keep='first')

        title_df_1['Color'] = 'firebrick'

        title_df_1['Size'] = 30

        title_df_1.columns = ['Words','Color','Size']

        

        # Combining all the dataframe together in one place



        frames = [title_df_1,word_df_prior_1,word_df_prior_strong_1,word_df_prior_newdata_1,word_df_speculative_1,word_df_unknown_1,word_df_novel_1]

         

        self.merged_df = pd.concat(frames)

        

        

    # Changing the index of merged_df to 'Words' so that can combine it with node



        self.merged_df_index = self.merged_df.drop_duplicates(subset='Words',keep='first')

        self.merged_df_index.set_index('Words',inplace=True)

        

        

    def network_creation(self):

        

        i = nx.Graph()

        

        



        if (self.title_df.empty ==False):

            for row in self.title_df.iterrows():

                i.add_edge(row[1]['Title1'], row[1]['Title1'], weight=row[1]['Score'])



        if (self.word_df_prior.empty ==False):

            for row in self.word_df_prior.iterrows():

                i.add_edge(row[1]['title'], row[1]['prior'], weight=row[1]['Weight'])



        if (self.word_df_prior_strong.empty ==False):

            for row in self.word_df_prior_strong.iterrows():

                i.add_edge(row[1]['title'], row[1]['prior-strong'], weight=row[1]['Weight'])

            

        if (self.word_df_novel.empty ==False):

            for row in self.word_df_novel.iterrows():

                i.add_edge(row[1]['title'], row[1]['novel'], weight=row[1]['Weight'])

                

        if (self.word_df_speculative.empty ==False):

            for row in self.word_df_speculative.iterrows():

                i.add_edge(row[1]['title'], row[1]['speculative'], weight=row[1]['Weight'])

                

        if (self.word_df_prior_newdata.empty ==False):

            for row in self.word_df_prior_newdata.iterrows():

                i.add_edge(row[1]['title'], row[1]['prior_newdata'], weight=row[1]['Weight'])

                

        if (self.word_df_unknown.empty ==False):

            for row in self.word_df_unknown.iterrows():

                i.add_edge(row[1]['title'], row[1]['unknown'], weight=row[1]['Weight'])

            

        merged_df_clr = self.merged_df_index.reindex(i.nodes())



        merged_df_clr['Color']=pd.Categorical(merged_df_clr['Color'])



        merged_df_clr['Color'].cat.codes

        

        

        #Plotting the force directed graph

        

        plt.figure(figsize=(22, 22))

        degrees = nx.degree(i)

        pos_node = nx.spring_layout(i,k=0.5)

        nx.draw_networkx(i,pos=pos_node,node_color=merged_df_clr['Color'].cat.codes, cmap=plt.cm.Set2,node_size=[(degrees[v] + 1) * 100 for v in i.nodes()],alpha = 0.7)

        

    def force_directed_graphs(self):

        

        net = Network(notebook=True,height = '720px',width = '1200px')

#         self.path = os.path.dirname(__file__) + "/templates/template.html"

        

        temp1 = self.merged_df

        temp2 = temp1.reset_index()



        temp2.drop(columns=['index'],inplace=True)



        net.add_nodes(temp2['Words'],title = temp2['Words'],color=temp2['Color'],size=temp2['Size'].to_list())



        if (self.title_df.empty ==False):

            for row in self.title_df.iterrows():

                net.add_edge(row[1]['Title1'], row[1]['Title1'], weight=row[1]['Score'])



        if (self.word_df_prior.empty ==False):

            for row in self.word_df_prior.iterrows():

                net.add_edge(row[1]['title'], row[1]['prior'], weight=row[1]['Weight'])



        if (self.word_df_prior_strong.empty ==False):

            for row in self.word_df_prior_strong.iterrows():

                net.add_edge(row[1]['title'], row[1]['prior-strong'], weight=row[1]['Weight'])

            

        if (self.word_df_novel.empty ==False):

            for row in self.word_df_novel.iterrows():

                net.add_edge(row[1]['title'], row[1]['novel'], weight=row[1]['Weight'])

                

        if (self.word_df_speculative.empty ==False):

            for row in self.word_df_speculative.iterrows():

                net.add_edge(row[1]['title'], row[1]['speculative'], weight=row[1]['Weight'])

                

        if (self.word_df_prior_newdata.empty ==False):

            for row in self.word_df_prior_newdata.iterrows():

                net.add_edge(row[1]['title'], row[1]['prior_newdata'], weight=row[1]['Weight'])

                

        if (self.word_df_unknown.empty ==False):

            for row in self.word_df_unknown.iterrows():

                net.add_edge(row[1]['title'], row[1]['unknown'], weight=row[1]['Weight'])



        net.save_graph("mygraph2.html")

        return net

#         display(net.show("mygraph.html"))
#Calling all the functions

ng= network_graph(task_answers)

ng.extract_titles(task_answers)

ng.extract_words(task_answers)

a = ng.force_directed_graphs()

a.show("mygraph2.html")