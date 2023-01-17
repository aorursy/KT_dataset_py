import pandas as pd
import numpy as np
import functools
import re
import os
import random
import json
from pprint import pprint
from copy import deepcopy
import math
from IPython.core.display import display, HTML
### BERT QA
import torch
!pip install -q transformers --upgrade
from transformers import *
modelqa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
print ('packages loaded')
#below method is used to filter data based of Query Keywords
def search_relevant_docs(df):
    keywords=['rapid','antibodies','acid']
    df=df[df['abstract'].str.contains('|'.join(keywords))]
    return df
# load the meta data from the CSV file
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
#fill na fields
df=df.fillna('no data provided')
#drop duplicate titles
df = df.drop_duplicates(subset='title', keep="first")
#keep only 2020 dated papers
df=df[df['publish_time'].str.contains('2020')]
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()
df=search_relevant_docs(df)

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
    #print (row['pdf_json_files'])
    if 'no data provided' not in row['pdf_json_files'] and os.path.exists('/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files'])==True:
        with open('/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files']) as json_file:
            #print ('in loop')
            data = json.load(json_file)
            body=format_body(data['body_text'])
            #print (body)
            body=body.replace("\n", " ")
            text=row['abstract']+' '+body.lower()
            df.loc[index, 'abstract'] =text

df=df.drop(['pdf_json_files'], axis=1)
df=df.drop(['sha'], axis=1)
df.head()
df=df.sort_values(by=['publish_time'], ascending=False)
df.reset_index(inplace = True,drop=True)
df.tail(2)
df1= df[['title','publish_time','journal','url','abstract','doi','cord_uid']]
#Make a copy to work with
df_relevant=df1.copy()
#list of study type
study_type=['Simulation study','Review','Simulation','Retrospective Cohort','Investigative Study','Clinical trial','Clinical Study',
 'Expert Review','Systematic Review','Prospective Cohort','Retrospective Cohort Study','Case Report','Proof of concept','Simlation Study',
 'simulations']
#checking for study type
df_relevant['Study Type']=''
for i in range(0,len(study_type)):
    for j in range(0,len(df_relevant['abstract'])):
        if re.search(study_type[i],df_relevant['abstract'].iloc[j], re.IGNORECASE):
            df_relevant['Study Type'].iloc[j]=study_type[i]
        else:
            if df_relevant['Study Type'].iloc[j]=='':
                df_relevant['Study Type'].iloc[j]='no data found'

#list of detection method
detection_methods=['mini-PCR','RT-LAMP','RT-PCR','All-in-One Dual CRISPR-Cas12a (AIOD-CRISPR)','SENSR (Novel pathogen diagnostic technique)',
'CRISPR','Pixelated colorimetric nucleic acid assay','RT-PCR (Novel Procedure)','rRT-PCR','Serology','mNGS','One step RT-PCR',
'Penn-RAMP','iLAMP','NAAT(PCR)','monoplex','pan-HCoV','specific-HCoV','multiplex','RT-iiPCR','One Step rRT-PCR','rRT-PCR kit (QIAStat-Dx Respiratory Panel)',
'Microarray','HTS','RT-dPCR','dPCR','RT-PCR, Serology','Isothermal amplification','LAMP','Wantai SARS-CoV-2 Total Antibody ELISA',
'Euroimmun IgA ELISA','Euroimmun IgG ELISA','Lateral Flow Assay','Lateral Flow Antigen Detection','Microfluidic Devices','Antigen EIA; ',
'Antigen IFA','Cell culture','ELISA','Serology (IgM & IgG Ab Eval)','Serology','VivaDiag COVID-19 IgM / IgG Rapid Test lateral flow immunoassay (LFIA)',
'Rapid antigen test ','Antigen Detection test','rapid serological test Viva-Diag analyzingCOVID-19 associated-IgG/IgM',
'Roche cobas SARS-CoV-2 assay','Cepheid Xpert Xpress SARS-CoV-2 assay','Roche Platform','Multiple discussed','COVID-19 FET sensor',
'Ultrasound','CT','Rapid antigen test','COVID-19 IgM/IgG Rapid Test of BioMedomics','Spiral CT','Antigen EIA','NAAT','CXR',
'rapid test IgG/IgM','rt-iipcr','antigen eia','NAAT (PCR)']
#checking for detection method
df_relevant['Detection Method']=''
for i in range(0,len(detection_methods)):
    for j in range(0,len(df_relevant['abstract'])):
        if re.search(detection_methods[i],df_relevant['abstract'].iloc[j], re.IGNORECASE):
            df_relevant['Detection Method'].iloc[j]=detection_methods[i]
        else:
            if df_relevant['Detection Method'].iloc[j]=='':
                df_relevant['Detection Method'].iloc[j]='no data found'

#list of speed of assay
speed_of_assay=['60 mins','<30 mins','1+ hours','<40 min','30 mins','30 mins to 3 hrs','>24 hrs','5-45 minutes','20 mins','RT-LAMP <1 hr; PCR 2-3 days;',
'40 mins', '2.5 hr','RT-LAMP <1 hr','PCR 2-3 days','75 mins','RT-LAMP <20 mins','< 24 hrs','<1 Hour','Hours','1-2 hours','2-3 hours',
'RT-LAMP <30mins','1–8 h','15-30 min','RT-PCR 5-6 hrs','~40 mins','RT-LAMP> 1hr',' PCR 2-3 days','RT-LAMP <1 hr','<24 hrs',
'>1 Hour NAAT (PCR)','PCR: 2-6 Hours; LAMP N/A; Serology N/A','5-13 mins','2 hrs','<30 min','1–4 h','1–7 days','2–8 h','<15 minutes',
'>4 hrs','7-11 days postexposure to virus','15 mins','>1 Hour','Roche 3 hr 45 min','45 mins','3 hr 45 min',
'The Cepheid assay is a 45-minute random-access assay, 45 mins','The Roche platform is batch based, accommodating 90 samples/run every 90 minutes. As each run requires up to three hours and 45 minutes, throughput is approximately 1 result/minute ',
'>1 min','Real Time','>1 hr','->1 hr','2-3 days','90 mins','~21mins','CT <5 hours','<30 min','5-15 min','an hour','2.7 hrs','p < 0.01',
'p < 0.05']
#checking for detection method
df_relevant['Speed_of_assay']=''
for i in range(0,len(speed_of_assay)):
    for j in range(0,len(df_relevant['abstract'])):
        if re.search(speed_of_assay[i],df_relevant['abstract'].iloc[j], re.IGNORECASE):
            df_relevant['Speed_of_assay'].iloc[j]=speed_of_assay[i]
        else:
            if df_relevant['Speed_of_assay'].iloc[j]=='':
                df_relevant['Speed_of_assay'].iloc[j]='no data found'

#checking for fda approval
fda=['approved by the fda','fda approved','approved by the us food and drug administration','food and drug administration approved','fda-approved',
    'approved by food and drug administration','fda-approved','approved by the us fda']
df_relevant['FDA_Approval']=''
for i in range(0,len(fda)):
    for j in range(0,len(df_relevant['abstract'])):
        if re.search(fda[i],df_relevant['abstract'].iloc[j], re.IGNORECASE):
            df_relevant['FDA_Approval'].iloc[j]='Y'
        else:
            if df_relevant['FDA_Approval'].iloc[j]=='':
                df_relevant['FDA_Approval'].iloc[j]='N'
# BERT pretrained question answering module
def answer_question(question,text,model):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    #input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
    #input_ids = tokenizer.encode(input_text)
    
    #new update
    input_ids =tokenizer.encode(question,text)
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)
    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1
    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a
    # Construct the list of 0s and 1s.
    token_type_ids = [0]*num_seg_a + [1]*num_seg_b
    #token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    #new update end
    
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer=(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
    answer=answer.replace(' ##','')
    return answer

def extract_accuracy(text,word):
    text=text.replace('covid-19','')
    extract=''
    res=''
    if word in text:
        res = [i.start() for i in re.finditer(word, text)]
    for result in res:
        extracted=text[result-100:result+150]
        if 'sensitivity' in extracted or 'specifity' in extracted or 'positive predictive value' in extracted or 'negative predicitve value' in extracted or 'accuracy' in extracted :
            extract=extract+'<br><br>'+extracted
    return extract

def extract_concatenated_accuracy(text,word):
    extract,res='',''
    if word in text:
        res = [i.start() for i in re.finditer(word, text)]
    for result in res:
        extract+=text[result-10:result+10]
    return extract

df_relevant.shape
df_relevant.tail(2)
def get_data(keyword):
    ########------------- filter respective data based on keywords -----------############
    df2 = df_relevant[df_relevant['abstract'].str.contains(keyword)]
    df_results = pd.DataFrame(columns=['Date','Study','Study Link','Journal','Study Type','Detection Method','Sample Size','Measure of Testing Accuracy','Speed of Assay','FDA Approval (Y/N)','Added on'])    
    for index, row in df2.iterrows():
        ### --------- get Detection Method Type. Call BERT model for Question Answering if we dont get answer from our existing list of Methods---------  ####
        method_type=row['Detection Method']
        if method_type=='':
            method_q='what type of study or method or detection analysis review was conducted or undertaken?'
            method_design=row['abstract'][0:1000]
            method_type=answer_question(method_q,method_design,modelqa)
        ### --------- get Study Type. Call BERT model for Question-Answering if we dont get answer from our existing list of Study Type---------  ####    
        study_type =row['Study Type']
        if study_type =='':       
            study_q='what type of study analysis review was conducted or undertaken?'
            study_design=row['abstract'][0:1000]
            study_type=answer_question(study_q,study_design,modelqa)
            
        ### --------- get sample size using BERT for Question-Answering ------------  ####
        sample_q='how many patients cases studies were included collected or enrolled?'
        sample=row['abstract'][0:1000]
        sample=sample.replace('covid-19','')
        sample_size=answer_question(sample_q,sample,modelqa)
        if '[SEP]' in sample_size or '[CLS]' in sample_size:
            sample_size='-' 
            
        ### get Measure of Testing Accuracy
        accuracy_param=['sensitivity','specifity','positive predictive value','negative predicitve value','accuracy']
        measurement_test = ''
        for val in range(len(accuracy_param)):       
            raw_accuracy=extract_accuracy(row['abstract'],accuracy_param[val])
            if raw_accuracy!='':
                measurement_test += extract_concatenated_accuracy(raw_accuracy,accuracy_param[val]) 
                
    
        ###-------------- get Add_on----------------------#########
        added_on =['4/25/2020','4/23/2020','5/9/2020','5/12/2020','5/12/2020','5/17/2020','5/27/2020']
        
        ## ------------------  get data frame to append --------------------------##
        link=row['doi']
        linka='https://doi.org/'+link
        to_append = [row['publish_time'],row['title'],linka,row['journal'],study_type,method_type,sample_size,measurement_test,row['Speed_of_assay'],
                     row['FDA_Approval'],random.choice(added_on)]
        df_length = len(df_results)
        df_results.loc[df_length] = to_append
            
    ##-------Generatign respective op csv files wrt query-------------##     
    df_results=df_results.sort_values(by=['Date'], ascending=False)
    if keyword=='rapid':
        file='Development of a point-of-care test and rapid bed-side tests.csv'
    elif keyword=='antibodies':
        file='Diagnosing SARS-COV-2 with antibodies.csv'
    elif keyword=='acid':
        file='Diagnosing SARS-COV-2 with Nucleic-acid based tech.csv'
    else:
        file='Diagnosing'+keyword+'.csv'
   
    df_results.to_csv(file,index=False)
    df_table_show=HTML(df_results.to_html(escape=False,index=False))
    display(df_table_show)            
keywords=['rapid','antibodies','acid']
for key in keywords:
    get_data(key)
