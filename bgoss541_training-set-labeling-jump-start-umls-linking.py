from glob import glob

import json

import pandas as pd

from tqdm.notebook import tqdm



dir_list = [

    '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv',

    '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset',

    '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license',

    '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset'

]



results_list = list()

for target_dir in dir_list:

    

    print(target_dir)

    

    for json_fp in tqdm(glob(target_dir + '/*.json')):



        with open(json_fp) as json_file:

            target_json = json.load(json_file)



        data_dict = dict()

        data_dict['doc_id'] = target_json['paper_id']

        data_dict['title'] = target_json['metadata']['title']



        abstract_section = str()

        for element in target_json['abstract']:

            abstract_section += element['text'] + ' '

        data_dict['abstract'] = abstract_section



        full_text_section = str()

        for element in target_json['body_text']:

            full_text_section += element['text'] + ' '

        data_dict['full_text'] = full_text_section

        

        results_list.append(data_dict)

    

df_results = pd.DataFrame(results_list)

df_results
df_results.to_csv('covid_text_20200322.csv', index=False)
!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
import spacy

import scispacy

import en_core_sci_sm

from scispacy.umls_linking import UmlsEntityLinker



nlp = en_core_sci_sm.load()
from scispacy.abbreviation import AbbreviationDetector



linker = UmlsEntityLinker(resolve_abbreviations=True)

abbreviation_pipe = AbbreviationDetector(nlp)



nlp.add_pipe(abbreviation_pipe)

nlp.add_pipe(linker)

df_abx = df_results.loc[df_results['abstract'] != '']

df_abx
umls_dict = dict()  # Track CUIs and scores

umls_df = dict()  # Track document frequency

classify_dict = dict()  # Track indexing per document



for row in tqdm(df_abx.itertuples(), total=df_abx.shape[0]):

       

    if row.abstract is not None:

        

        doc = nlp(row.abstract)   



        umls_set = set()

        

        classify_dict[row.doc_id] = dict()

        

        for entity in doc.ents:

            

            for umls_entity in entity._.umls_ents:

                

                if umls_entity[0] not in umls_dict:

                    umls_dict[umls_entity[0]] = [umls_entity[1]]

                else:

                    umls_dict[umls_entity[0]].append(umls_entity[1])

                    

                umls_set.add(umls_entity[0])

                

                if umls_entity[0] not in classify_dict[row.doc_id]:

                    classify_dict[row.doc_id][umls_entity[0]] = [umls_entity[1]]

                else:

                    classify_dict[row.doc_id][umls_entity[0]].append(umls_entity[1])

                    

        for entity in umls_set:



            if entity not in umls_df:

                umls_df[entity] = 1

            else:

                umls_df[entity] += 1

    
from statistics import mean 

import math



umls_results = list()



umls_idf_lookup = dict()



for cui, scores in tqdm(umls_dict.items()):

    

    umls_idf_lookup[cui] = {

        'total_count': len(scores),

        'idf': math.log(df_abx.shape[0] / umls_df[cui]),  # No smoothing? Entire population known?

        'average_score': mean(scores)

    }
results_list_tfidf = list()

df_tfidf = pd.DataFrame()



for doc_id, classify_data in tqdm(classify_dict.items()):

    

    doc_list = list()

    radicand = 0

    

    if len(classify_data) == 0:

        continue

    

    for cui, scores in classify_data.items():

                

        doc_list.append({

            'cui': cui,

            'tf': len(scores),

            'idf': umls_idf_lookup[cui]['idf'],

            'doc_id': doc_id

        })

        

        radicand += len(scores) ** 2

        

    denominator = math.sqrt(radicand)

    

    for x in doc_list:

        x['tf_idf'] = (x['tf'] * x['idf']) / denominator

        results_list_tfidf.append(x)

    

df_tfidf = pd.DataFrame(results_list_tfidf)

df_tfidf

# Give each term its name



df_tfidf['canonical_name'] = df_tfidf['cui'].apply(lambda x: linker.umls.cui_to_entity[x].canonical_name)
# Sort values, save, and display in notebook



df_tfidf.sort_values(by=['doc_id', 'tf_idf'], ascending=False)

df_tfidf = df_tfidf[['doc_id', 'canonical_name', 'cui', 'tf_idf', 'tf', 'idf']]

df_tfidf.to_csv('tfidf_named_20200322.csv', index=False)

df_tfidf
# Check our work



target_doc_id = df_tfidf['doc_id'].sample(n=1).values[0]

print(df_results.loc[df_results['doc_id'] == target_doc_id, 'abstract'].values[0])

df_tfidf.loc[df_tfidf['doc_id'] == target_doc_id].sort_values(by='tf_idf', ascending=False).head(10)

# Creating a resource list of the concepts used in this approach



df_concepts = df_tfidf.drop_duplicates(subset=['cui']).copy()

df_concepts = df_concepts[['cui', 'canonical_name', 'idf']]

df_concepts['definition'] = df_concepts['cui'].apply(lambda x: linker.umls.cui_to_entity[x].definition)

df_concepts['raw_count'] = df_concepts['cui'].apply(lambda x: umls_idf_lookup[x]['total_count'])

df_concepts['average_score'] = df_concepts['cui'].apply(lambda x: umls_idf_lookup[x]['average_score'])



df_concepts.to_csv('concepts_20200322.csv', index=False)

df_concepts.sort_values(by='average_score')

df_tfidf.query("canonical_name == 'RNA Processing'").sort_values(by='tf', ascending=False)
df_concepts.loc[df_concepts['definition'].str.contains('coronav', case=False, na=False)]
import plotly_express as px



fig = px.histogram(df_tfidf, x='tf_idf', title='tf-idf scores')

fig.show()
fig = px.scatter(df_concepts, 

                 x='idf', 

                 y='average_score',

                 size='raw_count',

                 hover_name="canonical_name")

fig.show()