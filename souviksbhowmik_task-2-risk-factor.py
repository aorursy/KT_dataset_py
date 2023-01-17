

import nltk

from nltk import word_tokenize

from nltk.util import ngrams

from collections import Counter

from nltk.book import FreqDist

from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize 
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import os

from tqdm.notebook import tqdm



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import string
#!ls CORD-19-research-challenge
working_list=[]
path='/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'



for file in tqdm(os.listdir(path)):

    #print(file)

    text_content = ''

    working_dict = dict()

    content = json.load(open(path+file,'r'))

    #print(type(content))

    working_dict['paper_id'] = content['paper_id']

    working_dict['path'] = path

    if 'metadata' in content and 'title' in content['metadata']:

        working_dict['title'] = content['metadata']['title']

        #print('found title')

        text_content = text_content  +' '+ content['metadata']['title'].lower()

    if 'abstract' in content:

        #print('found abstract')

        for abst in content['abstract'] :

            text_content = text_content+' '+abst['text'].lower()

            

    if 'body_text' in content:

        #print('found body text')

        for bt in content['body_text'] :

            text_content = text_content+' '+bt['text'].lower()

    

    

    working_dict['text'] = text_content

    working_list.append(working_dict)

    
path='/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'



for file in tqdm(os.listdir(path)):

    #print(file)

    text_content = ''

    working_dict = dict()

    content = json.load(open(path+file,'r'))

    #print(type(content))

    working_dict['paper_id'] = content['paper_id']

    working_dict['path'] = path

    if 'metadata' in content and 'title' in content['metadata']:

        working_dict['title'] = content['metadata']['title']

        #print('found title')

        text_content = text_content  +' '+ content['metadata']['title'].lower()

    if 'abstract' in content:

        #print('found abstract')

        for abst in content['abstract'] :

            text_content = text_content+' '+abst['text'].lower()

            

    if 'body_text' in content:

        #print('found body text')

        for bt in content['body_text'] :

            text_content = text_content+' '+bt['text'].lower()

    

    

    working_dict['text'] = text_content

    working_list.append(working_dict)
len(working_list)
working_df = pd.DataFrame(working_list)
working_df.head()
def clean_tokens(token_temp):

    token = []

    for t in token_temp:

        temp = t

        for s in string.punctuation:

            temp=temp.replace(s,'')

        if len(temp.strip())>0:

            token.append(t)

    return token
def list_to_str_tokens(list_val):

    s=''

    for l in list_val:

        s=s+','+l

    return s
#This would create better tokenization ..but is resource intensive

#working_df['token_list'] = working_df['text'].apply(lambda x:nltk.word_tokenize(x))

#working_df['token_list'] = working_df['token_list'].apply(clean_tokens)

#working_df['token_text']=working_df['token_list'].apply(list_to_str_tokens)

# ideally would have preferred upto 3 ngrams

#vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=30000)

vectorizer = TfidfVectorizer(analyzer='word')

#max_features
#count_vector = vectorizer.fit_transform(doc_df['token_text'])

tfidf_vector = vectorizer.fit_transform(working_df['text'])
task_df=pd.read_csv('/kaggle/input/covidtask2/Task2.csv',header=None)
search_string = list_to_str_tokens(list(task_df[0]))
search_string = search_string+' coronavirus'+' corona virus'+' covid19'+' covid-19 '+'covid 19'
search_string=search_string.lower()
search_string
# search_string = 'transmission, incubation, environmental stability ,'\

#     +'natural history, transmission, and diagnostics for the virus'\

#             +'What is known about transmission, incubation, and environmental stability? '\

#             +'What do we know about natural history, transmission, and diagnostics for '\

#             +'the virus? What have we learned about infection prevention and control?'\

#             +'Specifically, we want to know what the literature reports about:'\

#             +'Range of incubation periods for the disease in humans' \

#             +'(and how this varies across age and health status) and '\

#             +'how long individuals are contagious, even after recovery.'\

#             +'Prevalence of asymptomatic shedding and transmission' \

#             +'(e.g., particularly children).'\

#             +'Seasonality of transmission.'\

#             +'Physical science of the coronavirus '\

#             +'(e.g., charge distribution, adhesion to hydrophilic/phobic surfaces,' \

#              +'environmental survival to inform decontamination efforts for affected areas '\

#              +'and provide information about viral shedding).'\

#             +'Persistence and stability on a multitude of substrates and sources '\

#             +'(e.g., nasal discharge, sputum, urine, fecal matter, blood).'\

#             +'Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).'\

#             +'Natural history of the virus and shedding of it from an infected person'\

#             +'Implementation of diagnostics and products to improve clinical processes'\

#             +'Disease models, including animal models for infection, disease and transmission'\

#             +'Tools and studies to monitor phenotypic change and potential adaptation of the virus'\

#             +'Immune response and immunity'\

#             +'Effectiveness of movement control strategies to prevent secondary'+'transmission in health care and community settings'\

#             +'Effectiveness of personal protective equipment (PPE) and its usefulness to '\

#             +'reduce risk of transmission in'+'health care and community settings'\

#             +'Role of the environment in transmission'\

#             + 'Corona virus Coronavirus covid covid-19 covid19'
query_vector = vectorizer.transform([search_string])
similarity_list = []

dense_query_vector = query_vector.todense()

for i in tqdm(range(tfidf_vector.shape[0])):

    dense_doc_vector = tfidf_vector[i].todense()

    #two_vec=np.concatenate((query_vector, tfidf_vector[i]), axis=0)

    two_vec=np.concatenate((dense_query_vector, dense_doc_vector), axis=0)

    similarity_list.append(cosine_similarity(two_vec)[0,1])

    #break

    
working_df['query_1_score']=similarity_list
working_df_sorted = working_df.sort_values(['query_1_score'],ascending=False).reset_index()
working_df_sorted.head(15)
top_search_df=working_df_sorted.head(50).drop_duplicates(subset=['query_1_score'],keep='first')
#working_list[10269]
#working_list[11213]
print('Top Response')

print(top_search_df.loc[0]['paper_id'])

print(top_search_df.loc[0]['title'])

print(top_search_df.loc[0]['path'])

#working_df_sorted.head(5).loc[0]['text']
top_search_df.head(15)
detail_list=[]

for index in tqdm(range(top_search_df.shape[0])):

    row= top_search_df.iloc[index]

    path = top_search_df.iloc[index]['path']

    paper_id =  top_search_df.iloc[index]['paper_id']

    #print(path)

    file_name=path+paper_id+'.json'

    #print(paper_id)

    

    content = json.load(open(file_name,'r'))

    if 'metadata' in content and 'title' in content['metadata']:

        title=content['metadata']['title']

        #print(title)

    

    if 'abstract' in content:

        #print('found abstract')

        for abst in content['abstract'] :

            detail_dict={}

            detail_dict['paper_id']=paper_id

            detail_dict['path']=path

            detail_dict['title']=title

            detail_dict['text']=abst['text'].lower()

            detail_dict['content_type']='abstract'

            detail_list.append(detail_dict)

            

    if 'body_text' in content:

        #print('found body text')

        count = 0

        for bt in content['body_text'] :

            count = count+1

            detail_dict={}

            detail_dict['paper_id']=paper_id

            detail_dict['path']=path

            detail_dict['title']=title

            detail_dict['text']=bt['text'].lower()

            detail_dict['content_type']='paragraph '+str(count)

            detail_list.append(detail_dict)
detail_df = pd.DataFrame(detail_list)

detail_df.shape,detail_df['paper_id'].nunique()
detail_df.head()
detail_df['token_list'] = detail_df['text'].apply(lambda x:nltk.word_tokenize(x))

detail_df['token_list'] = detail_df['token_list'].apply(clean_tokens)

detail_df['token_text']=detail_df['token_list'].apply(list_to_str_tokens)
detail_df.head(10)
detail_vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,3))

#detail_vectorizer = TfidfVectorizer(analyzer='word')
tfidf_detail_vector = detail_vectorizer.fit_transform(detail_df['token_text'])
tfidf_detail_vector.shape
question_list=list(task_df[0])
question_list
count=0



for question in tqdm(question_list):

    count=count+1

    ques = question.lower()

    ques_tokens = nltk.word_tokenize(ques)

    ques_token_list = clean_tokens(ques_tokens)

    question_token_text = list_to_str_tokens(ques_token_list)

    question_vector = detail_vectorizer.transform([question_token_text])

    question_similarity_list = []

    dense_question_vector = question_vector.todense()

    #similarity_list=[]

    for i in range(tfidf_detail_vector.shape[0]):

        dense_doc_vector = tfidf_detail_vector[i].todense()

        #two_vec=np.concatenate((query_vector, tfidf_vector[i]), axis=0)

        two_vec=np.concatenate((dense_question_vector, dense_doc_vector), axis=0)

        question_similarity_list.append(cosine_similarity(two_vec)[0,1])

    #count=count+1

    detail_df['question_'+str(count)+'_similarity']=question_similarity_list
detail_df.head()
count=0

for question in question_list:

    count=count+1

    print('Question\n\n')

    print(question)

    print('--------------------------------------------------------')

    temp_df = detail_df.sort_values(['question_'+str(count)+'_similarity'],ascending=False).reset_index()

    print('Top search ',temp_df.loc[0]['paper_id'])

    print('title',temp_df.loc[0]['title'])

    print('section',temp_df.loc[0]['content_type'])

    print(temp_df.loc[0]['text'])

    print('\n')

    print('Second best search ',temp_df.loc[1]['paper_id'])

    print('title',temp_df.loc[1]['title'])

    print('section',temp_df.loc[1]['content_type'])

    print(temp_df.loc[1]['text'])

    print('\n\n')

    