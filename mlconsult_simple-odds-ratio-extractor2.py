import pandas as pd

import numpy as np

import functools

import re

print ('packages loaded')
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

print ('ALL CORD19 articles',df.shape)

#fill na fields

df=df.fillna('no data provided')

#drop duplicate titles

df = df.drop_duplicates(subset='title', keep="first")

#keep only 2020 dated papers

df=df[df['publish_time'].str.contains('2020')]

# convert abstracts to lowercase

df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()

#show 5 lines of the new dataframe

df=search_focus(df)

print ('Keep only COVID-19 related articles',df.shape)



import os

import json

from pprint import pprint

from copy import deepcopy

import math





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

            text=row['abstract']+' '+body.lower()



            df.loc[index, 'abstract'] =text



df=df.drop(['full_text_file'], axis=1)

df=df.drop(['sha'], axis=1)

df.head()
focus='hypertension'

df1 = df[df['abstract'].str.contains(focus)]

print (focus,'focused articles',df1.shape)
from IPython.core.display import display, HTML



def extract_ratios(text,word):

    extract=''

    if word in text:

        res = [i.start() for i in re.finditer(word, text)]

    for result in res:

        extracted=text[result:result+75]

        #print (extracted)

        #if '95' in extracted or 'odds ratio' in extracted or 'p>' in extracted or '=' in extracted or 'p<' in extracted or '])' in extracted or '(rr' in extracted:

        if '95%' in extracted or 'odds ratio' in extracted or '])' in extracted or '(rr' in extracted or '(ar' in extracted or '(hr' in extracted or '(or' in extracted:

            extract=extract+' '+extracted

    #print (extract)

    return extract



focus='hypertension'

df_results = pd.DataFrame(columns=['date','study','link','extracted'])

for index, row in df1.iterrows():

    extracted=extract_ratios(row['abstract'],focus)

    if extracted!='':

        link=row['doi']

        linka='https://doi.org/'+link

        to_append = [row['publish_time'],row['title'],linka,extracted]

        df_length = len(df_results)

        df_results.loc[df_length] = to_append



df_results=df_results.sort_values(by=['date'], ascending=False)

df_table_show=HTML(df_results.to_html(escape=False,index=False))

display(df_table_show)
