

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import json

import re

import scipy as sc

import warnings



import matplotlib.pyplot as plt

import os



root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()

meta_df.info()
meta_df.head()

!pip install -U sentence-transformers


from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer



warnings.filterwarnings("ignore")

model = SentenceTransformer('bert-base-nli-mean-tokens')
import glob

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
all_json[0]
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            

            self.paper_id = content['paper_id']

            

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

def get_breaks(content, length):

    data = ""

    words = content.split(' ')

    total_chars = 0



    # add break every length characters

    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data
dict_ = {'publish_time': [], 'url': [], 'paper_id': [], 'doi':[], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    

    try:

        content = FileReader(entry)

    except Exception as e:

        continue  # invalid paper format, skip



    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['abstract'].append(content.abstract)

    dict_['paper_id'].append(content.paper_id)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 100 words append with ...

        info = content.abstract.split(' ')[:100]

        summary = get_breaks(' '.join(info), 40)

        dict_['abstract_summary'].append(summary + "...")

    else:

        # abstract is short enough

        summary = get_breaks(content.abstract, 40)

        dict_['abstract_summary'].append(summary)

        

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    

    try:

        # if more than one author

        authors = meta_data['authors'].values[0].split(';')

        if len(authors) > 2:

            # if more than 2 authors, take them all with html tag breaks in between

            dict_['authors'].append(get_breaks('. '.join(authors), 40))

        else:

            # authors will fit in plot

            dict_['authors'].append(". ".join(authors))

    except Exception as e:

        # if only one author - or Null valie

        dict_['authors'].append(meta_data['authors'].values[0])

    

    # add the title information, add breaks when needed

    try:

        title = get_breaks(meta_data['title'].values[0], 40)

        dict_['title'].append(title)

    # if title was not provided

    except Exception as e:

        dict_['title'].append(meta_data['title'].values[0])

    

    # add meta_data information

    dict_['journal'].append(meta_data['journal'].values[0])

    dict_['doi'].append(meta_data['doi'].values[0])

    dict_['publish_time'].append(meta_data['publish_time'].values[0])

    dict_['url'].append(meta_data['url'].values[0])

    

    

df_covid = pd.DataFrame(dict_, columns=['publish_time', 'url', 'paper_id', 'doi', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])

df_covid.head()
df_covid.info() 

df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)

df_covid['abstract'].describe(include='all')
df_covid.describe()
df_covid.head()
df_covid['abstract'].describe(include='all')

df = df_covid
!pip install langdetect
from tqdm import tqdm

from langdetect import detect

from langdetect import DetectorFactory



# set seed

DetectorFactory.seed = 1000000007



# hold label - language

languages = []



# go through each text

for ii in tqdm(range(0,len(df))):

    # split by space into list, take the first x intex, join with space

    text = df.iloc[ii]['body_text'].split(" ")

    

    lang = "en"

    try:

        if len(text) > 50:

            lang = detect(" ".join(text[:50]))

        elif len(text) > 0:

            lang = detect(" ".join(text[:len(text)]))

    # ught... beginning of the document was not in a good format

    except Exception as e:

        all_words = set(text)

        try:

            lang = detect(" ".join(all_words))

        # what!! :( let's see if we can find any text in abstract...

        except Exception as e:

            

            try:

                # let's try to label it through the abstract then

                lang = detect(df.iloc[ii]['abstract_summary'])

            except Exception as e:

                lang = "unknown"

                pass

    

    # get the language    

    languages.append(lang)
from pprint import pprint



languages_dict = {}

for lang in set(languages):

    languages_dict[lang] = languages.count(lang)

    

print("Total: {}\n".format(len(languages)))

pprint(languages_dict)
df['language'] = languages

plt.bar(range(len(languages_dict)), list(languages_dict.values()), align='center')

plt.xticks(range(len(languages_dict)), list(languages_dict.keys()))

plt.title("Distribution of Languages in Dataset")

plt.show()
df = df[df['language'] == 'en'] 

df.info()

df.head()
df.dropna(inplace=True)

df.head()

df.info()

task1="""Specifically, we want to know what the literature reports about:



Length of viral shedding after illness onset

Incubation period across different age groups

What is the Incubation Period of the Virus?

Proportion of patients who were asymptomatic

Proportion of pediatric COVID19 patients who were asymptomatic

Asymptomatic transmission during incubation

Natural history of the virus from an infected person

What is the median viral shedding duration?

What is the longest duration of viral shedding?

Manifestations of COVID-19 including but not limited to possible cardiomyopathy and cardiac arrest

How does viral load relate to disease presentations and likelihood of a positive diagnostic test_.csv

What do we know about disease models?"""
task=[task1]

query=[]

for i in range(len(task)):

    task[i]=task[i].split("\n")

    query.append(task[i][2:])

print(query)
query_embeddings=[]

for i in range(len(query)):

    query_embeddings.append(model.encode(query[i]))
df.reset_index(drop = True, inplace = True)

df['abstract_summary']

#abstract_embeddings = model.encode(df['abstract'])

abstract_summary_embeddings = model.encode(df['abstract_summary'])

def getfile_insensitive(path):

    directory, filename = os.path.split(path)

    directory, filename = (directory or '.'), filename.lower()

    for f in os.listdir(directory):

        newpath = os.path.join(directory, f)

        if os.path.isfile(newpath) and f.lower() == filename:

            return newpath



def isfile_insensitive(path):

    return getfile_insensitive(path) is not None



sample_document='3_patient_descriptions'

scores=[]

def formatting(_topdf):

    _topdf.rename(columns={'journal':'Journal'},inplace=True)

    _topdf.rename(columns={'url':'Study Link'},inplace=True)

    _topdf.rename(columns={'publish_time':'Date'},inplace=True)

    _topdf.rename(columns={'title':'Study'},inplace=True)

    

    return _topdf

for tsk in range(len(task)):

    

    for prob, query_embedding in zip(query[tsk], query_embeddings[tsk]):

        dis = sc.spatial.distance.cdist([query_embedding], abstract_summary_embeddings, "cosine")[0]

        print(dis)

        results = zip(range(len(dis)), dis)

        results = sorted(results, key=lambda x: x[1])

        print("Query:", prob)

        print("Answer:" )

        scores.append(1-results[0][1])

        print(df['abstract'][results[0][0]].strip(), "\n(Score: %.4f)" % (1-results[0][1]),"\n")

        k=10

        print(results[:k])

        topk=results[:k]

        id,id_v=zip(*topk)

        topData=df.iloc[1]

        _topdf = df.iloc[list(id), :]

        _topdf = formatting(_topdf)

        csv_str=prob.replace('?','_')+'.csv'

        print(csv_str)

        path1 = f'{root_path}Kaggle/target_tables/{sample_document}/'+csv_str

        path1 = f'{path1}'

        print(path1)

        if (isfile_insensitive(path1)):

            path1=getfile_insensitive(path1)

            q_df = pd.read_csv(path1)

        else:      

            path1 = f'{root_path}Kaggle/target_tables/{sample_document}/'

            path1 = os.path.join(path1,os.listdir(path1)[-1])

            q_df = pd.read_csv(path1)

            q_df.drop(q_df.index,inplace=True)

        #print('../input/CORD-19-research-challenge/Kaggle/target_tables/3_patient_descriptions/What is the incubation period of the virus_.csv')

        #print(path1)

        q_df.head()

        q_df

        #print(len(q_df))

        #print(len(_topdf))

        #print(q_df.columns)

        res_df = pd.merge(q_df, _topdf, how='outer', on=['Study','Study Link','Date','Journal'])[q_df.columns[1:]]

        #print(len(res_df))

        res_df.info()

        res_csv=r'/kaggle/output/'+ csv_str

        print(res_csv)

        res_df.to_csv( csv_str, index = True)