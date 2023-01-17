import functools

from IPython.core.display import display, HTML

from nltk import PorterStemmer

import torch

from transformers import *



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

import numpy as np

import pandas as pd



# keep only docsuments with covid -cov-2 and cov2

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

print (df.shape)

#drop duplicates

#df=df.drop_duplicates()

#drop NANs 

df=df.fillna('no data provided')

df = df.drop_duplicates(subset='title', keep="first")

# convert abstracts to lowercase

df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()

df = df[df['publish_time'].str.contains('2020')]

#show 5 lines of the new dataframe

df=search_focus(df)

print (df.shape)

df.head()
import os

import json

from pprint import pprint

from copy import deepcopy

import math

from IPython.core.display import display, HTML



def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        #print (section)

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    #print('_______')

    return body



df['body']=''

df['method_results']=''

for index, row in df.iterrows():

    if os.path.exists('/kaggle/input/CORD-19-research-challenge/'+row['full_text_file']+'/'+row['full_text_file']+'/pdf_json/'+row['sha']+'.json'):

        with open('/kaggle/input/CORD-19-research-challenge/'+row['full_text_file']+'/'+row['full_text_file']+'/pdf_json/'+row['sha']+'.json') as json_file:

            data = json.load(json_file)

            body=format_body(data['body_text'])

            #print (body)

            body=body.replace("\n", "")

            body=body.replace(",", "")

            body=body.lower()

            df.loc[index, 'body'] = body

            if body!='' and 'method' in body and 'results' in body:

                temp=body.split('method')[1]

                method=temp.split('results')[0]

                df.loc[index, 'method_results'] = method



df=df[df['body'].str.contains('severe')]

df=df[df['body'].str.contains('risk')]

print (df.shape)

df.head()
#tell the system how many sentences are needed

max_sentences=5



# function to stem keywords into a common base word

def stem_words(words):

    stemmer = PorterStemmer()

    singles=[]

    for w in words:

        singles.append(stemmer.stem(w))

    return singles



# BERT pretrained question answering module

def answer_question(question,text,tokenizer,model):

    input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"

    input_ids = tokenizer.encode(input_text)

    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    #print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    answer=(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    # show qeustion and text

    #tokenizer.decode(input_ids)

    return answer





# list of lists for topic words realting to tasks

display(HTML('<h1>COVID-19 Risk Factors</h1>'))

#display(HTML('<h3>Table of Contents (ctrl f to search for the hash tagged words below to find that data table)</h3>'))

#tasks = [['comorbidities','comorbid'],['risk factor','risk factors'],['cancer patient', 'cancer patients'],['hypertension','hyperten'],['heart', 'disease'],['chronic', 'bronchitis'],['cerebral', 'infarction'],['diabetes', 'diabete'],['copd','copd'],["blood type","type"],['smoking','smok'],['basic','reproductive','number'],["incubation", "period", "days"]]

#tasks = ['hypertension','diabetes','heart disease','male','copd','smoking','age','stroke','cerbrovascular','cancer','respiratory','kidney disease','drinking','tuberculosis','bmi']

tasks = ['age']

z=0

for terms in tasks:

    stra=' '

    stra=' '.join(terms)

    k=str(z)

    z=z+1

# loop through the list of lists

z=0

for search_words in tasks:

    df_table = pd.DataFrame(columns = ["pub_date","title","link","journal","severe","sever sig.","severe age adj.","Severe OR Calculated or Extracted","fatality","fatality sig.","fatality age adj.","Fatality OR Calculated or Extracted","design","sample","study pop.","risk factor"])

    str1=''

    # a make a string of the search words to print readable search

    str1=search_words

    df1=df[df['body'].str.contains(search_words)]

    

    display(HTML('<h3>Task Topic: '+str1+'</h3>'))

    #display(HTML('# '+str1+' <a></a>'))

    z=z+1

    # record how many sentences have been saved for display

    # loop through the result of the dataframe search

    for index, row in df1.iterrows():

        odds=''

        pub_sentence=''

        sentences_used=0

        #break apart the absracrt to sentence level

        sentences = row['body'].split('. ')

        #loop through the sentences of the abstract

        for sentence in sentences:

            # missing lets the system know if all the words are in the sentence

            missing=0

            #loop through the words of sentence

            for word in search_words:

                #if keyword missing change missing variable

                if word not in sentence:

                    missing=missing+1

            if tasks[0][0] in sentence and 'or=' in sentence or tasks[0][0] in sentence and 'hr=' in sentence or tasks[0][0] in sentence and 'rr=' in sentence or tasks[0][0] in sentence and 'aor=' in sentence or tasks[0][0] in sentence and 'ahr=' in sentence:

                odds=odds+'...'+sentence

            # after all sentences processed show the sentences not missing keywords limit to max_sentences

            if missing < len(search_words)-1 and sentences_used < max_sentences and len(sentence)<1000 and sentence!='':

                sentence=sentence.capitalize()

                if sentence[len(sentence)-1]!='.':

                    sentence=sentence+'.'

                pub_sentence=pub_sentence+'<br><br>'+sentence

        if pub_sentence!='':

            sentence=pub_sentence

            sentences_used=sentences_used+1

            if row['method_results']!='':

                text=row['method_results'][ 0 : 1000 ]

            else:

                text=row['abstract'][ 0 : 1000 ]

            

            question='how many patients or cases were in the study, review or analysis?'

            sample=answer_question(question,text,tokenizer,model)

            sample=sample.replace("#", "")

            sample=sample.replace(" , ", ",")

            if sample=='19' or sample=='' or '[SEP]'in sample:

                sample='unk'

            if len(sample)>50:

                sample='unk'

            sample=sample.replace(" ", "")

            

            question='what type or kind of review or study was conducted?'

            design=answer_question(question,text,tokenizer,model)

            design=design.replace(" ##", "")

            if '[SEP]'in design or '[CLS]' in design or len(design)>75:

                design='unk'

            

            question='what is the name of the hospital or country?'

            study_pop=answer_question(question,text,tokenizer,model)

            study_pop=study_pop.replace(" ##", "")

            if '[SEP]'in study_pop:

                study_pop='unk'

                

            #question='How many severe outcomes?'

            #severe=answer_question(question,text,tokenizer,model)

            #severe=severe.replace(" ##", "")

            #severe=severe.replace(" . ", ".")

            #severe=severe.replace(" · ", ".")

            #if '[SEP]'in severe:

                #severe='unk'

            

            #question='How many fatalities of deaths?'

            #fatality=answer_question(question,text,tokenizer,model)

            #fatality=fatality.replace(" ##", "")

            #fatality=fatality.replace(" . ", ".")

            #fatality=fatality.replace(" · ", ".")

            #if '[SEP]'in fatality:

                #fatality='unk'

            

            

            

            #question='what was the or= with '+tasks[0][0]+'?'

            #calc=answer_question(question,text,tokenizer,model)

            

            authors=row["authors"].split(" ")

            link=row['doi']

            title=row["title"]

            linka='https://doi.org/'+link

            linkb=title

            journal=row['journal']

            if journal=='no data provided':

                journal=row['full_text_file']

            sentence='<p align="left">'+sentence+'</p>'

            final_link='<p align="left"><a href="{}">{}</a></p>'.format(linka,linkb)

            #to_append = [row['publish_time'],title,linka,row['journal'],severe,'ss','saa','sOR','fatality','fs','faa','fOR',design,sample,study_pop]

            to_append = [row['publish_time'],title,linka,journal,odds,'-','-','-','-','-','-','-',design,sample,study_pop,search_words]

            df_length = len(df_table)

            df_table.loc[df_length] = to_append

    filename=str1+'.csv'

    df_table=df_table.sort_values('pub_date', ascending=False)

    df_table.to_csv(filename,index = False)

        #display(HTML('<b>'+sentence+'</b> - <i>'+title+'</i>, '+'<a href="https://doi.org/'+link+'" target=blank>'+authors[0]+' et al.</a>'))

    df_table=HTML(df_table.to_html(escape=False,index=False))

    display(df_table)

print ("done")