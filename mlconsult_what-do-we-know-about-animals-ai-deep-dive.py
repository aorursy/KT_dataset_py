import spacy

nlp = spacy.load('en_core_web_lg')

import numpy as np

import pandas as pd

import re

import os

import json

from pprint import pprint

from copy import deepcopy

import math

import torch

!pip install rake-nltk

from rake_nltk import Rake

from nltk.corpus import stopwords

from rake_nltk import Metric, Rake
print ('python packages imported')



from IPython.core.display import display, HTML

# keep only documents with covid -cov-2 and cov2

def search_focus(df):

    dfa = df[df['abstract'].str.contains('covid')]

    dfb = df[df['abstract'].str.contains('-cov-2')]

    dfc = df[df['abstract'].str.contains('cov2')]

    dfd = df[df['abstract'].str.contains('ncov')]

    frames=[dfa,dfb,dfc,dfd]

    df = pd.concat(frames)

    df=df.drop_duplicates(subset='title', keep="first")

    return df



# load the meta data from the CSV file using 3 columns (abstract, title, authors),

df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','journal','abstract','authors','doi','publish_time','sha','full_text_file'])

print ('total documents ',df.shape)

#drop duplicates

#df=df.drop_duplicates()

#drop NANs 

df=df.fillna('no data provided')

df = df.drop_duplicates(subset='title', keep="first")

# convert abstracts to lowercase

df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()

#show 5 lines of the new dataframe

df=search_focus(df)

df = df[df['publish_time'].str.contains('2020')]

print ('COVID-19 focused documents ',df.shape)

df.head()



def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body





for index, row in df.iterrows():

    if ';' not in row['sha'] and os.path.exists('/kaggle/input/CORD-19-research-challenge/'+row['full_text_file']+'/'+row['full_text_file']+'/pdf_json/'+row['sha']+'.json')==True:

        with open('/kaggle/input/CORD-19-research-challenge/'+row['full_text_file']+'/'+row['full_text_file']+'/pdf_json/'+row['sha']+'.json') as json_file:

            data = json.load(json_file)

            body=format_body(data['body_text'])

            keyword_list=['TB','incidence','age']

            #print (body)

            body=body.replace("\n", " ")



            df.loc[index, 'abstract'] = body.lower()



df=df.drop(['full_text_file'], axis=1)

df=df.drop(['sha'], axis=1)

df.head()



# add full text back after testing
import functools

def search_focus_shape(df,focus):

    df1 = df[df['abstract'].str.contains(focus)]

    #df1=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in focus))]

    return df1



focus_term='animal'

df1=search_focus_shape(df,focus_term)

print ('focus term: ',focus_term)

print ('# focused papers',df1.shape)
r = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,min_length=2, max_length=2) # Uses stopwords for english from NLTK, and all puntuation characters.Please note that "hello" is not included in the list of stopwords.



def extract_data(text,word):

    extract=''

    if word in text:

        #text = re.sub(r'[^\w\s]','',text)

        res = [i.start() for i in re.finditer(word, text)]

        after=text[res[0]:res[0]+15]

        before=text[res[0]-15:res[0]]

        raw = before+after

        parts=raw.split()

        parts = parts[1:-1]

        extract= ' '.join(parts)

        extract=extract.replace('animals','animal')

    return extract

text=''

for index, row in df1.iterrows():

    extracted=extract_data(row['abstract'],focus_term)

    if extracted!='':

        text=text+' '+extracted

a=r.extract_keywords_from_text(text)

term_list=r.get_ranked_phrases()

term_list = sorted(term_list, key=str.lower)

#c=r.get_ranked_phrases_with_scores()

print(term_list)

#print(c)

print('___________________')



# custom sentence score

def score_sentence_prob(search,sentence,focus):

    final_score=0

    keywords=search.split()

    sent_parts=sentence.split()

    word_match=0

    missing=0

    for word in keywords:

        word_count=sent_parts.count(word)

        word_match=word_match+word_count

        if word_count==0:

            missing=missing+1

    percent = 1-(missing/len(keywords))

    final_score=abs((word_match/len(sent_parts)) * percent)

    if missing==0:

        final_score=final_score+.05

    if focus in sentence:

        final_score=final_score+.05

    return final_score



def score_docs(df,focus,search):

    df_results = pd.DataFrame(columns=['date','study','link','excerpt','score'])

    df1=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in focus))]

    master_text=''

    for index, row in df1.iterrows():

        pub_sentence=''

        sentences=row['abstract'].split('.')

        hi_score=0

        for sentence in sentences:

            if len(sentence)>75 and search in sentence:

                rel_score=score_sentence_prob(search,sentence,focus)

                #rel_score=score_sentence(search,sentence)

                if rel_score>.0002:

                    #print (sentence,rel_score)

                    pub_sentence=pub_sentence+' '+sentence+' '+str(round(rel_score,2))

                    if rel_score>hi_score:

                        hi_score=rel_score

                    master_text=master_text+' '+pub_sentence

        if pub_sentence!='':

            #print (row['abstract'])

            #print ('------------------')

            link=row['doi']

            linka='https://doi.org/'+link

            to_append = [row['publish_time'],row['title'],linka,pub_sentence,hi_score]

            df_length = len(df_results)

            df_results.loc[df_length] = to_append

    df_results=df_results.sort_values(by=['date'], ascending=False)



    return df_results

for term in term_list:

    if focus_term in term and any(map(str.isdigit, term))==False and ')' not in term:

        df_results=score_docs(df,focus_term,term)

        if df_results.empty==False:

            print (term)

            df_table_show=HTML(df_results.to_html(escape=False,index=False))

            display(df_table_show)