import pandas as pd

import numpy as np

import functools

import re

### BERT QA

import torch

!pip install -q transformers --upgrade

from transformers import *

modelqa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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



# load the meta data from the CSV file

#usecols=['title','journal','abstract','authors','doi','publish_time','sha','full_text_file']

df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

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



def format_tables(ref_entries):

    extract=''

    for x in ref_entries.values():

        if 'html' in x:

            start = '<html>'

            end = '</html>'

            x=str(x).lower()

            dat=(x.split(start))[1].split(end)[0]

            extract=extract+' '+dat

    

    return extract



df['tables']='N/A'

for index, row in df.iterrows():

    #print (row['pdf_json_files'])

    if 'no data provided' not in row['pdf_json_files'] and os.path.exists('/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files'])==True:

        with open('/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files']) as json_file:

            #print ('in loop')

            data = json.load(json_file)

            body=format_body(data['body_text'])

            #ref_entries=format_tables(data['ref_entries'])

            #print (body)

            body=body.replace("\n", " ")

            text=row['abstract']+' '+body.lower()

            df.loc[index, 'abstract'] = text

            #df.loc[index, 'tables'] = ref_entries



df=df.drop(['pdf_json_files'], axis=1)

df=df.drop(['sha'], axis=1)

df.tail()
from IPython.core.display import display, HTML

import string



def extract_ratios(text,word):

    extract=''

    if word in text:

        res = [i.start() for i in re.finditer(word, text)]

    for result in res:

        extracted=text[result:result+75]

        extracted2=text[result-200:result+75]

        level=''

        if 'sever' in extracted2:

            level=level+' severe'

        if 'fatal' in extracted2:

            level=level+' fatal'

        if 'death' in extracted2:

            level=level+' fatal'

        if 'mortality' in extracted2:

            level=level+' fatal'

        if 'hospital' in extracted2:

            level=level+' severe'

        if 'intensive' in extracted2:

            level=level+' severe'

        #print (extracted)

        #if '95' in extracted or 'odds ratio' in extracted or 'p>' in extracted or '=' in extracted or 'p<' in extracted or '])' in extracted or '(rr' in extracted:

        if 'odds ratio' in extracted or '])' in extracted or '(rr' in extracted or '(ar' in extracted or '(hr' in extracted or '(or' in extracted or '(aor' in extracted or '(ahr' in extracted:

            if '95%' in extracted:

                extract=extract+' '+extracted+' '+level

    #print (extract)

    return extract



def get_ratio(text):

    char1 = '('

    char2 = '95%'

    ratio=text[text.find(char1)+1 : text.find(char2)]

    ratio=ratio.replace('â','')

    return ratio



# get the upper and lower bounds from the extracted data

def get_bounds(text):

    raw=''

    char1 = 'ci'

    char2 = ')'

    data=text[text.find(char1)+1 : text.find(char2)]

    

    if '-' in data:

        raw=data.split('-')

        low=raw[0][-5:]

        hi=raw[1][:5]

    

    if 'to' in data and raw=='':

        raw=data.split('to')

        low=raw[0][-5:]

        hi=raw[1][:5]

        

    if ',' in data and raw=='':

        raw=data.split(',')

        low=raw[0][-5:]

        hi=raw[1][:5]

    

    if raw=='':

        return '-','-'

    low=low.replace('·','.')

    low = re.sub("[^0-9.]+", "", low)

        

    return low,hi



# get the p value fomr the extracted text

def get_pvalue(text):

    raw=''

    pvalue=''

    char1 = 'ci'

    char2 = ')'

    data=text[text.find(char1)+1 : text.find(char2)]

    

    if 'p=' in data:

        raw=data.split('p=')

        pvalue='p='+raw[1][:7]

        

    if 'p =' in data:

        raw=data.split('p =')

        pvalue='p='+raw[1][:7]

    

    if 'p>' in data and raw=='':

        raw=data.split('p>')

        pvalue='p>'+raw[1][:7]

        

    if 'p<' in data and raw=='':

        raw=data.split('p<')

        pvalue='p<'+raw[1][:7]

    

    if pvalue=='':

        return '-'

    pvalue=pvalue.replace('â','')

    return pvalue



# extract study design

def extract_design(text):

    words=['retrospective','prospective cohort','retrospective cohort', 'systematic review',' meta ',' search ','case control','case series,','time series','cross-sectional','observational cohort', 'retrospective clinical','virological analysis','prevalence study','literature','two-center']

    study_types=['retrospective','prospective cohort','retrospective cohort','systematic review','meta-analysis','literature search','case control','case series','time series analysis','cross sectional','observational cohort study', 'retrospective clinical studies','virological analysis','prevalence study','literature search','two-center']

    extract=''

    res=''

    for word in words:

        if word in text:

            res = [i.start() for i in re.finditer(word, text)]

        for result in res:

            extracted=text[result-30:result+30]

            extract=extract+' '+extracted

    i=0

    study=''

    for word in words:

        if word in extract:

            study=study_types[i]

        #print (extract)

        i=i+1

    return study



# BERT pretrained question answering module

def answer_question(question,text,model,tokenizer):

    input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"

    input_ids = tokenizer.encode(input_text)

    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    #print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    answer=(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    # show qeustion and text

    #tokenizer.decode(input_ids)

    answer=answer.replace(' ##','')

    #print (answer)

    return answer



def process_text(df1,focus):

    significant=''

    #df_results = pd.DataFrame(columns=['date','study','link','extracted','ratio','lower bound','upper bound','significant','p-value'])

    df_results = pd.DataFrame(columns=['Date', 'Study', 'Study Link', 'Journal', 'Study Type', 'Severity of Disease', 'Severity lower bound', 'Severity upper bound', 'Severity p-value', 'Severe significance', 'Severe adjusted', 'Hand-calculated Severe', 'Fatality', 'Fatality lower bound', 'Fatality upper bound', 'Fatality p-value', 'Fatality significance', 'Fatality adjusted', 'Hand-calculated Fatality', 'Multivariate adjustment', 'Sample size', 'Study population', 'Critical Only', 'Discharged vs. Death', 'Added on', 'DOI', 'CORD_UID'])

    for index, row in df1.iterrows():

        study_type=''

        study_type=extract_design(row['abstract'])

        extracted=extract_ratios(row['abstract'],focus)

        if extracted!='':

            ratio=get_ratio(extracted)

            lower_bound,upper_bound=get_bounds(extracted)

            if lower_bound!='-' and lower_bound!='':

                if float(lower_bound)>1:

                    significant='yes'

                else:

                    significant='no'

            else:

                significant='-'

            pvalue=get_pvalue(extracted)

            

            if 'aor' in extracted or 'arr' in extracted or 'ahr' in extracted or 'arr' in extracted or 'adjusted' in extracted:

                adjusted='yes'

            else: adjusted='no'

            

            if 'fatal' in extracted and 'severe' not in extracted:

                severe='-'

                slb='-'

                sub='-'

                spv='-'

                ss='-'

                sa='-'

                fatal=ratio

                flb=lower_bound

                fub=upper_bound

                fpv=pvalue

                fs=significant

                fa=adjusted

            else:

                fatal='-'

                flb='-'

                fub='-'

                fpv='-'

                fs='-'

                fa='-'

                severe=ratio

                slb=lower_bound

                sub=upper_bound

                spv=pvalue

                ss=significant

                sa=adjusted

            

            ### get sample size

            sample_q='how many patients cases studies were included collected or enrolled'

            sample=row['abstract'][0:1000]

            sample_size=answer_question(sample_q,sample,modelqa,tokenizer)

            if '[SEP]' in sample_size or '[CLS]' in sample_size:

                sample_size='-'

            sample_size=sample_size.replace(' , ',',')

                

            link=row['doi']

            linka='https://doi.org/'+link

            #to_append = [row['publish_time'],row['title'],linka,extracted,ratio,lower_bound,upper_bound,significant,pvalue]

            to_append = [row['publish_time'], row['title'], linka, row['journal'], study_type, severe, slb, sub, spv, ss, sa, '-', fatal, flb, fub, fpv, fs, fa, '-', '-', sample_size, '-', '-', '-', '-', row['doi'], row['cord_uid']]

            df_length = len(df_results)

            df_results.loc[df_length] = to_append

    return df_results



focuses=['hypertension','diabetes','male','gender','heart disease', 'copd','smok',' age ','cerebrovascular','cardiovascular disease','cancer','kidney disease','respiratory disease','drinking','obes','liver disease']



for focus in focuses:

    df1 = df[df['abstract'].str.contains(focus)]

    df_results=process_text(df1,focus)

    df_results=df_results.sort_values(by=['Date'], ascending=False)

    df_table_show=HTML(df_results.to_html(escape=False,index=False))

    display(HTML('<h1> Risk Factor '+focus+'</h1>'))

    display(df_table_show)

    file=focus+'.csv'

    df_results.to_csv(file,index=False)
