###################### LOAD PACKAGES ##########################

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk import PorterStemmer

from IPython.core.display import display, HTML

import pandas as pd

import numpy as np

import functools



import spacy

nlp = spacy.load('en_core_web_lg')

!pip install bert-extractive-summarizer

from summarizer import Summarizer



import torch

from transformers import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

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

print (df.shape)

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

print (df.shape)

df.head()



df_master=pd.read_csv('../input/aipowered-literature-review-csvs/kaggle/working/Risk Factors/Heart Disease.csv')

#usecols=["Date","Study","Study Link","Journal","Days","Range (Days)","Sample (n)","Study Type"]

df_master.drop(df_master.columns[df_master.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

df_master.head()
# BERT pretrained question answering module

def answer_question(question,text, model,tokenizer):

    input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"

    input_ids = tokenizer.encode(input_text)

    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    #print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    answer=(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    # show qeustion and text

    #tokenizer.decode(input_ids)

    answer=answer.replace(" ##", "")

    answer=answer.replace(" · ", "·")

    answer=answer.replace(" . ", ".")

    answer=answer.replace(" , ", ",")

    if '[SEP]'in answer or '[CLS]' in answer or answer=='':

        answer='unk'

        

    return answer



# custom sentence score

def score_sentence_prob(search,sentence,focus):

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



def process_question(df,search,focus,df_master):

    df_table = pd.DataFrame(data=None, columns=df_master.columns)

    # focuses to make sure the exact phrase in text

    #df1 = df[df['abstract'].str.contains(focus)]

    # focus to make sure all words in text

    df1=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in focus))]

    for index, row in df1.iterrows():

        sentences = row['abstract'].split('. ')

        pub_sentence=''

        hi_score=0

        study=''

        hi_study_score=0

        for sentence in sentences:

            if len(sentence)>75 and focus in sentence:

                rel_score=score_sentence_prob(search,sentence,focus)

                if rel_score>.05:

                    sentence=sentence.capitalize()

                    if sentence[len(sentence)-1]!='.':

                        sentence=sentence+'.'

                    pub_sentence=pub_sentence+' '+sentence

                    if rel_score>hi_score:

                        hi_score=rel_score

                

        if pub_sentence!='':

            text=row['abstract'][0:1000]

            

            question='how many patients or cases were in the study, review or analysis?'

            sample=answer_question(question,text,model,tokenizer)

            sample=sample.replace("#", "")

            sample=sample.replace(" , ", ",")

            if sample=='19' or sample=='' or '[SEP]'in sample:

                sample='unk'

            if len(sample)>50:

                sample='unk'

            sample=sample.replace(" ", "")

            

            question='what type or kind of review study analysis model was used?'

            design=answer_question(question,text,model,tokenizer)

            design=design.replace(" ##", "")

            if '[SEP]'in design or '[CLS]' in design or design=='':

                design='unk'

            

            shorter = pub_sentence[0:1000]

            ### get sever numbers

            question='what is the '+focus+' HR OR RR AOR hazard odds ratio ()?'

            severe=answer_question(question,text,model,tokenizer)

            

            authors=row["authors"].split(" ")

            link=row['doi']

            title=row["title"]

            score=hi_score

            journal=row["journal"]

            if journal=='':

                journal=row['full_text_file']

            linka='https://doi.org/'+link

            linkb=title

            final_link='<p align="left"><a href="{}">{}</a></p>'.format(linka,linkb)

            #author_link='<p align="left"><a href="{}">{}</a></p>'.format(linka,authors[0]+' et al.')

            #sentence=pub_sentence+' '+author_link

            sentence=pub_sentence

            #sentence='<p fontsize=tiny" align="left">'+sentence+'</p>'

            #print(df_master.columns)

            to_append = [row['publish_time'],title,linka,journal,severe, '-',

       'Severe upper bound', 'Severe p-value', 'Severe Significant',

       '-', '-', '-',

       '-', '-', '-',

       '-', '-', '-',

       '-', design, sample, '-',

       '-', '-', '-']

            df_length = len(df_table)

            df_table.loc[df_length] = to_append

    df_table=df_table.sort_values(by=['Date'], ascending=False)

    to_append = df_master.columns

    df_length = len(df_table)

    df_table.loc[df_length] = to_append

    return df_table

###################### MAIN PROGRAM ###########################





### focus quesiton with single keyword

keywords = ['heart disease']

#keywords = ['hypertension','diabetes','heart disease','gender','copd','smoking','age','stroke','cerbrovascular','cancer','kidney disease','drinking','tuberculosis','obesity']

#'diabetes','heart disease','male gender','copd','smoking','age','stroke','cerbrovascular','cancer','kidney disease','drinking','tuberculosis','bmi'



q=0



# loop through the list of questions

for keyword in keywords:

    # limit results to severe risk factors

    search_words = 'severe risk factor heart disease death morbid fatal'

    

    # get best sentences

    df_table=process_question(df,search_words,keyword,df_master)

    df_answers=df_table

        

    display(HTML('<h3>'+search_words+'</h3>'))

    

    df_table_show = df_table.append(df_master,sort=False)

    df_table_show = df_table_show.drop_duplicates(subset='Title', keep="last")

    

    df_table_display=HTML(df_table_show.to_html(escape=False,index=True))

    display(df_table_display)

    

    df_table_show.to_csv('Heart Disease.csv', index = False)

print ('done')