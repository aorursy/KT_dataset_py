!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz
import os
import re
import ast
import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker

from nltk import sent_tokenize

from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

from numba import jit # parallel processing
nlp = spacy.load("/opt/conda/lib/python3.7/site-packages/en_ner_bc5cdr_md/en_ner_bc5cdr_md-0.2.4")
linker = UmlsEntityLinker(resolve_abbreviations=True)
nlp.add_pipe(linker)
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
nlp("test for the drug remdesivir for sars-cov-2 treatment and HCQ").ents
nlp("test for the drug azithromycin, lopinavir, remdesivir for sars-cov-2 treatment and HCQ").ents
input_data_path = '/kaggle/input/claims-flag-sentence-cord-uid-updated/'
input_data_file = 'claims_flag_sentence_cord_uid_updated.csv'

w2v_data_path= '/kaggle/input/claims-flag-each-cord-uid/'
w2v_data_file = 'claims_flag_each_cord_uid.csv'

metadata_file_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_file = 'metadata.csv'
metadata = pd.read_csv(metadata_file_path+metadata_file)
metadata = metadata[['cord_uid','publish_time','title']]
#input_data = pd.read_csv(input_data_path+input_data_file)
#input_processed = pd.read_csv(input_data_path+input_data_file)
input_data = pd.read_csv(input_data_path+input_data_file)
w2v_data = pd.read_csv(w2v_data_path+w2v_data_file)

input_data.cord_uid = input_data.cord_uid.str.lower().str.strip()
input_data.claims = input_data.claims.str.lower().str.strip()
w2v_data.cord_uid = w2v_data.cord_uid.str.lower().str.strip()
w2v_data.sentence = w2v_data.sentence.str.lower().str.strip()

input_processed_all = pd.merge(input_data,w2v_data[['cord_uid','sentence','w2vVector']],\
                            left_on=['cord_uid','claims'],right_on=['cord_uid','sentence'],how='left')
input_processed_all = input_processed_all.drop(columns='sentence_y')
input_processed_all.rename(columns={'sentence_x':'sentence'},inplace=True)
input_processed = input_processed_all.loc[~(input_processed_all.w2vVector.isnull()),:].reset_index()
claims_unmapped = input_processed_all.loc[(input_processed_all.w2vVector.isnull())\
                                          & input_processed_all.claims!='[]',:].reset_index()

print('No of unique sentences in claims:',input_data.loc[input_data.claims!='[]','claims'].nunique())
print('No of unique sentences in claims after join:',input_processed.claims.nunique())
print('No of rows in claims:',len(input_data.loc[input_data.claims!='[]',:]))
print('No of rows in claims after join:',len(input_processed))
print('No of rows in claims unmapped:',len(claims_unmapped))
print('\n')
print('No of unique papers in claims:',input_data.loc[input_data.claims!='[]','cord_uid'].nunique())
print('No of unique papers in claims after join:',input_processed.cord_uid.nunique())

print('\nPapers with no claims:',input_data.loc[input_data.claims=='[]','cord_uid'].nunique())
#For papers with no claims, tokenize to sentences and map to w2v vectors
no_claims = input_data.loc[input_data.claims=='[]',:].reset_index()

dict = {}
k = 0

for i,text in enumerate(no_claims.sentence):
    for sent in sent_tokenize(text):
        dict[i] = {'cord_uid':no_claims.cord_uid[i],\
                   'section':no_claims.section[i],\
                   'sentence':no_claims.sentence[i],\
                   'drug_terms_used':no_claims.drug_terms_used[i],\
                   'claims':sent}
        k = k + 1
        
no_claims = pd.DataFrame.from_dict(dict, "index")

no_claims_all = pd.merge(no_claims,w2v_data[['cord_uid','sentence','w2vVector']],\
                            left_on=['cord_uid','claims'],right_on=['cord_uid','sentence'],how='left')
no_claims_all = no_claims_all.drop(columns='sentence_y')
no_claims_all.rename(columns={'sentence_x':'sentence'},inplace=True)
no_claims_mapped = no_claims_all.loc[~(no_claims_all.w2vVector.isnull()),:].reset_index()
no_claims_unmapped = no_claims_all.loc[no_claims_all.w2vVector.isnull(),:].reset_index()

print('No. of rows mapped to vectors:', len(no_claims_mapped))
print('No. of rows unmapped to vectors:', len(no_claims_unmapped))
print('Total rows:',len(no_claims_all))
#Keep only sentences containing at least 3 words other than those defined below
#This also removes any sentences that do not contain any words at all

# rep = {"text": "", "cite_spans": "", "ref_spans": "", "section": "", "Abstract": "",\
#        "bioRxiv preprint": "", "medRxiv preprint": "", "doi:": ""}
# rep = dict((re.escape(k), v) for k, v in rep.items())
# pattern = re.compile("|".join(rep.keys()))
# sentences_temp = [pattern.sub(lambda m: rep[re.escape(m.group(0))], s) for s in input_data.sentence]
# pattern = re.compile(".*[A-Za-z].*")
# sentences_to_keep = [(bool(re.search(pattern,s))) & (len(s.split(' '))>2) for s in sentences_temp]
# input_processed = input_data.loc[sentences_to_keep,:]
# sentences_to_drop = [not i for i in sentences_to_keep]
# input_excluded = input_data.loc[sentences_to_drop,:]
#Convert w2vVector column from string type to  list
input_processed.w2vVector = [re.sub(',+', ',', ','.join(w.replace('\n','').split(' '))) for w in input_processed.w2vVector]
input_processed.w2vVector = [re.sub('\[,', '', w) for w in input_processed.w2vVector]
input_processed.w2vVector = [re.sub(',\]', '', w) for w in input_processed.w2vVector]
input_processed.w2vVector = [re.sub('\[', '', w) for w in input_processed.w2vVector]
input_processed.w2vVector = [re.sub('\]', '', w) for w in input_processed.w2vVector]
input_processed.w2vVector = input_processed.w2vVector.apply(lambda s: list(ast.literal_eval(s)))

no_claims_mapped.w2vVector = [re.sub(',+', ',', ','.join(w.replace('\n','').split(' '))) for w in no_claims_mapped.w2vVector]
no_claims_mapped.w2vVector = [re.sub('\[,', '', w) for w in no_claims_mapped.w2vVector]
no_claims_mapped.w2vVector = [re.sub(',\]', '', w) for w in no_claims_mapped.w2vVector]
no_claims_mapped.w2vVector = [re.sub('\[', '', w) for w in no_claims_mapped.w2vVector]
no_claims_mapped.w2vVector = [re.sub('\]', '', w) for w in no_claims_mapped.w2vVector]
no_claims_mapped.w2vVector = no_claims_mapped.w2vVector.apply(lambda s: list(ast.literal_eval(s)))
input_processed
no_claims_mapped
drug_terms = []
for drugs in input_data.drug_terms_used:
    drug_terms = drug_terms + drugs.split(',')
drug_terms = list(set(drug_terms))
drug_terms.append('acei/arb')
len(drug_terms)
#input_processed.to_csv('cord_drug_titles_abstracts_conclusions.csv')
#input_excluded.to_csv('cord_drug_titles_abstracts_conclusions_excluded.csv')
# title_data = input_processed.loc[input_processed.section=='title',:]
# abstract_data = input_processed.loc[input_processed.section=='abstract',:]
# title_abstract_data = input_processed.loc[(input_processed.section=='title') | (input_processed.section=='abstract'),:]
# conclusion_data = input_processed.loc[(input_processed.section!='title') & (input_processed.section!='abstract'),:]
# claims_data = input_processed.loc[input_processed.claim_flag==1,:]
# print('Number of papers:', input_processed.cord_uid.nunique())
# print('Number of papers with title:', title_data.cord_uid.nunique())
# print('Number of papers with abstract:', abstract_data.cord_uid.nunique())
# print('Number of papers with conclusion:', conclusion_data.cord_uid.nunique())

# print('\n')

# print('Number of papers with core claims:', claims_data.cord_uid.nunique())
# print('Number of papers with title in core claims:', claims_data.loc[claims_data.section=='title','cord_uid'].nunique())
# print('Number of papers with abstract in core claims:', claims_data.loc[claims_data.section=='abstract','cord_uid'].nunique())
# print('Number of papers with conclusion in core claims:', claims_data.loc[(claims_data.section!='title') & (claims_data.section!='abstract'),'cord_uid'].nunique())
# print('Number of unique sentences under titles:', title_data.sentence.nunique())
# print('Number of unique sentence ids under titles:', title_data.sentence_id.nunique())
# print('Number of unique sentences under abstracts:', abstract_data.sentence.nunique())
# print('Number of unique sentence ids under abstracts:', abstract_data.sentence_id.nunique())
# print('Number of unique sentences under claims:', claims_data.sentence.nunique())
# print('Number of unique sentence ids under claims:', claims_data.sentence_id.nunique())
# #Average w2v vectors of all sentences falling under a single cord_uid
# title_data_final = pd.DataFrame(columns = ['cord_uid','sentence','w2vVector','drugs'])
# for cord_uid in title_data.cord_uid.unique():
#     title = " ".join(title_data.loc[title_data.cord_uid==cord_uid,'sentence'])
#     drugs = ",".join(title_data.loc[title_data.cord_uid==cord_uid,'drug_terms_used'])
#     drugs = ",".join(list(set(drugs.split(','))))
#     w2vVector = np.mean(list(title_data.loc[title_data.cord_uid==cord_uid,'w2vVector']), axis=0)
#     title_data_final = title_data_final.append({'cord_uid':cord_uid,\
#                                                 'sentence': title,\
#                                                 'w2vVector': w2vVector,\
#                                                 'drugs': drugs},\
#                                               ignore_index=True)
# len(title_data_final)
# pattern = re.compile(".*hydroxychloroquine.*")
# sentences_to_keep = [(bool(re.search(pattern,s.lower()))) for s in title_data_final.sentence]
# drug_title_data = title_data_final.loc[sentences_to_keep,:].reset_index(drop=True)
# len(drug_title_data)
# title_similarity = pd.DataFrame(columns=['paper1_cord_uid','paper2_cord_uid','title1','title2','similarity_score'])
# jit(nopython=True, parallel=True)
# for i,paper1 in enumerate(drug_title_data.sentence):
#     for j,paper2 in enumerate(drug_title_data.sentence):
#         if i!=j:
#             cos_sim = cosine_similarity(drug_title_data.w2vVector[i].reshape(1,-1),drug_title_data.w2vVector[j].reshape(1,-1))[0][0]
#             title_similarity = title_similarity.append({'paper1_cord_uid':drug_title_data.cord_uid[i],\
#                                                         'paper2_cord_uid':drug_title_data.cord_uid[j],\
#                                                         'title1':paper1,\
#                                                         'title2':paper2,\
#                                                         'similarity_score':cos_sim},\
#                                                ignore_index=True)
# title_similarity = pd.DataFrame(columns=['paper1_cord_uid','paper2_cord_uid','title1','title2','drugs1','drugs2','similarity_score'])
# jit(nopython=True, parallel=True)
# title_pairs = list(combinations(title_data_final.index,2))
# for i,j in title_pairs:
#     drugs1 = title_data_final.drugs[i].split(',')
#     drugs2 = title_data_final.drugs[j].split(',')
#     if any(d1 in drugs2 for d1 in drugs1):
#         cos_sim = cosine_similarity(title_data_final.w2vVector[i].reshape(1,-1),title_data_final.w2vVector[j].reshape(1,-1))[0][0]
#         title_similarity = title_similarity.append({'paper1_cord_uid':title_data_final.cord_uid[i],\
#                                                     'paper2_cord_uid':title_data_final.cord_uid[j],\
#                                                     'title1':title_data_final.sentence[i],\
#                                                     'title2':title_data_final.sentence[j],\
#                                                     'similarity_score':cos_sim,\
#                                                     'drugs1':title_data_final.drugs[i],\
#                                                     'drugs2':title_data_final.drugs[j]},\
#                                            ignore_index=True)

# title_similarity.to_csv('drug_title_similarity.csv')
# #Average w2v vectors of all sentences falling under a single cord_uid
# title_abstract_data_final = pd.DataFrame(columns = ['cord_uid','sentence','w2vVector','drugs'])
# for cord_uid in title_abstract_data.cord_uid.unique():
#     sentences = " ".join(title_abstract_data.loc[title_abstract_data.cord_uid==cord_uid,'sentence'])
#     drugs = ",".join(title_abstract_data.loc[title_abstract_data.cord_uid==cord_uid,'drug_terms_used'])
#     drugs = ",".join(list(set(drugs.split(','))))
#     w2vVector = np.mean(list(title_abstract_data.loc[title_abstract_data.cord_uid==cord_uid,'w2vVector']), axis=0)
#     title_abstract_data_final = title_abstract_data_final.append({'cord_uid':cord_uid,\
#                                                                 'sentence': sentences,\
#                                                                 'w2vVector': w2vVector,\
#                                                                  'drugs': drugs},\
#                                                                ignore_index=True)
# len(title_abstract_data_final)
# title_abstract_similarity = pd.DataFrame(columns=['paper1_cord_uid','paper2_cord_uid','text1','text2','drugs1','drugs2','similarity_score'])
# jit(nopython=True, parallel=True)
# paper_pairs = list(combinations(title_abstract_data_final.index,2))
# for i,j in paper_pairs:
#     drugs1 = title_abstract_data_final.drugs[i].split(',')
#     drugs2 = title_abstract_data_final.drugs[j].split(',')
#     if any(d1 in drugs2 for d1 in drugs1):
#         cos_sim = cosine_similarity(title_abstract_data_final.w2vVector[i].reshape(1,-1),title_abstract_data_final.w2vVector[j].reshape(1,-1))[0][0]
#         title_abstract_similarity = title_abstract_similarity.append({'paper1_cord_uid':title_abstract_data_final.cord_uid[i],\
#                                                     'paper2_cord_uid':title_abstract_data_final.cord_uid[j],\
#                                                     'text1':title_abstract_data_final.sentence[i],\
#                                                     'text2':title_abstract_data_final.sentence[j],\
#                                                     'similarity_score':cos_sim,\
#                                                     'drugs1':title_abstract_data_final.drugs[i],\
#                                                     'drugs2':title_abstract_data_final.drugs[j]},\
#                                            ignore_index=True)

# title_abstract_similarity.to_csv('drug_title_abstract_similarity.csv')
# #Average w2v vectors of all sentences falling under a single cord_uid
# claims_data_final = pd.DataFrame(columns = ['cord_uid','sentence','w2vVector','drugs'])
# for cord_uid in claims_data.cord_uid.unique():
#     sentences = " ".join(claims_data.loc[claims_data.cord_uid==cord_uid,'sentence'])
#     drugs = ",".join(claims_data.loc[claims_data.cord_uid==cord_uid,'drug_terms_used'])
#     drugs = ",".join(list(set(drugs.split(','))))
#     w2vVector = np.mean(list(claims_data.loc[claims_data.cord_uid==cord_uid,'w2vVector']), axis=0)
#     claims_data_final = claims_data_final.append({'cord_uid':cord_uid,\
#                                                                 'sentence': sentences,\
#                                                                 'w2vVector': w2vVector,\
#                                                                  'drugs': drugs},\
#                                                                ignore_index=True)
# len(claims_data_final)
# claims_similarity = pd.DataFrame(columns=['paper1_cord_uid','paper2_cord_uid','text1','text2','drugs1','drugs2','similarity_score'])
# jit(nopython=True, parallel=True)
# paper_pairs = list(combinations(claims_data_final.index,2))
# for i,j in paper_pairs:
#     drugs1 = claims_data_final.drugs[i].split(',')
#     drugs2 = claims_data_final.drugs[j].split(',')
#     if any(d1 in drugs2 for d1 in drugs1):
#         cos_sim = cosine_similarity(claims_data_final.w2vVector[i].reshape(1,-1),claims_data_final.w2vVector[j].reshape(1,-1))[0][0]
#         claims_similarity = claims_similarity.append({'paper1_cord_uid':claims_data_final.cord_uid[i],\
#                                                     'paper2_cord_uid':claims_data_final.cord_uid[j],\
#                                                     'text1':claims_data_final.sentence[i],\
#                                                     'text2':claims_data_final.sentence[j],\
#                                                     'similarity_score':cos_sim,\
#                                                     'drugs1':claims_data_final.drugs[i],\
#                                                     'drugs2':claims_data_final.drugs[j]},\
#                                            ignore_index=True)

# claims_similarity.to_csv('drug_claims_similarity.csv')
claims_data = input_processed
claims_data = claims_data.append(no_claims_mapped).reset_index(drop=True)

#Replace drug name short forms with full forms
dict = {'hcq':'hydroxychloroquine','cq':'chloroquine','azt':'azithromycin','azi':'azithromycin', 'az':'azithromycin'}
for key,value in dict.items():
    claims_data['claims'] = [t.lower().replace(key,value) for t in claims_data.claims]

# claims_similarity = pd.DataFrame(columns=['paper1_cord_uid','paper2_cord_uid','text1','text2','drugs1','drugs2','similarity_score'])
dict = {}
k = 0

print(claims_data.cord_uid.nunique())
print(len(claims_data))


jit(nopython=True, parallel=True)

paper_pairs = list(combinations(claims_data.index,2))
for i,j in paper_pairs:
    drugs1 = claims_data.drug_terms_used[i].split(',')
    drugs2 = claims_data.drug_terms_used[j].split(',')
    if any(d1 in drugs2 for d1 in drugs1) and (claims_data.cord_uid[i]!=claims_data.cord_uid[j]):
        cos_sim = cosine_similarity(np.array(claims_data.w2vVector[i]).reshape(1,-1),np.array(claims_data.w2vVector[j]).reshape(1,-1))[0][0]
        dict[k] = {'paper1_cord_uid':claims_data.cord_uid[i],\
                    'paper2_cord_uid':claims_data.cord_uid[j],\
                    'text1':claims_data.claims[i],\
                    'text2':claims_data.claims[j],\
                    'similarity_score':cos_sim,\
                    'drugs1':claims_data.drug_terms_used[i],\
                    'drugs2':claims_data.drug_terms_used[j]}
        k = k + 1
claims_similarity = pd.DataFrame.from_dict(dict, "index")
# Claims with no vectors
claims_data = claims_unmapped
claims_data = claims_data.append(no_claims_unmapped).reset_index(drop=True)

#Replace drug name short forms with full forms
dict = {'hcq':'hydroxychloroquine','cq':'chloroquine','azt':'azithromycin','azi':'azithromycin', 'az':'azithromycin'}
for key,value in dict.items():
    claims_data['claims'] = [t.lower().replace(key,value) for t in claims_data.claims]

print(claims_data.cord_uid.nunique())
print(len(claims_data))

dict = {}
k = 0

jit(nopython=True, parallel=True)

paper_pairs = list(combinations(claims_data.index,2))
for i,j in paper_pairs:
    drugs1 = claims_data.drug_terms_used[i].split(',')
    drugs2 = claims_data.drug_terms_used[j].split(',')
    if any(d1 in drugs2 for d1 in drugs1) and (claims_data.cord_uid[i]!=claims_data.cord_uid[j]):
        dict[k] = {'paper1_cord_uid':claims_data.cord_uid[i],\
                    'paper2_cord_uid':claims_data.cord_uid[j],\
                    'text1':claims_data.claims[i],\
                    'text2':claims_data.claims[j],\
                    'similarity_score':'NA',\
                    'drugs1':claims_data.drug_terms_used[i],\
                    'drugs2':claims_data.drug_terms_used[j]}
        k = k + 1
claims_with_no_vectors = pd.DataFrame.from_dict(dict, "index")
print(len(claims_similarity))
print(len(claims_with_no_vectors))
print(claims_similarity.paper1_cord_uid.nunique())
print(claims_similarity.paper2_cord_uid.nunique())
print(claims_with_no_vectors.paper1_cord_uid.nunique())
print(claims_with_no_vectors.paper2_cord_uid.nunique())
claims_all = claims_similarity.loc[claims_similarity.similarity_score >=0.5,:]
claims_all = claims_all.append(claims_with_no_vectors).reset_index(drop=True)
print('After filtering on similarity scores:',len(claims_all))

k = 0
dict = {}

for i in range(0,len(claims_all)):
#     drugs1 = list(nlp(claims_all.loc[i,'text1']).ents)
#     drugs2 = list(nlp(claims_all.loc[j,'text2']).ents)
    drugs1 = [d for d in drug_terms if d in claims_all.text1[i]]
    drugs2 = [d for d in drug_terms if d in claims_all.text2[i]]
    if any(d1 in drugs2 for d1 in drugs1):
        dict[k] = {'paper1_cord_uid':claims_all.paper1_cord_uid[i],\
                    'paper2_cord_uid':claims_all.paper2_cord_uid[i],\
                    'text1':claims_all.text1[i],\
                    'text2':claims_all.text2[i],\
                    'similarity_score':claims_all.similarity_score[i],\
                    'drugs1':drugs1,\
                    'drugs2':drugs2}
        k = k + 1
claims_filtered = pd.DataFrame.from_dict(dict, "index")

print('After filtering on drug names:',len(claims_filtered))
print(claims_filtered.paper1_cord_uid.nunique())
print(claims_filtered.paper2_cord_uid.nunique())
claims_filtered = pd.merge(claims_filtered,metadata,how='inner',\
                           left_on = 'paper1_cord_uid',\
                          right_on = 'cord_uid')
cols_rename = {'title':'title1','publish_time':'publish_time1'}
claims_filtered.rename(columns = cols_rename,inplace=True)

claims_filtered = pd.merge(claims_filtered,metadata,how='inner',\
                           left_on = 'paper2_cord_uid',\
                          right_on = 'cord_uid')
cols_rename = {'title':'title2','publish_time':'publish_time2'}
claims_filtered.rename(columns = cols_rename,inplace=True)

print(len(claims_filtered))
claims_filtered = claims_filtered.drop(columns=['cord_uid_x','cord_uid_y','similarity_score'])
claims_filtered = claims_filtered.drop_duplicates().reset_index(drop=True)
# claims_similarity.to_csv('drug_individual_claims_similarity.csv')
# claims_with_no_vectors.to_csv('drug_individual_claims_no_vectors.csv')
claims_filtered.to_csv('drug_individual_claims_filtered.csv')