#Here you define the drug you will search for
#examples: 
#research_drug='hydroxychloroquine'
##research_drug='remdesivir'
#research_drug='lopinavir'
#research_drug='ritonavir'
#research_drug='ribavirin'
#research_drug='naproxen'
#research_drug='clarithromycin'
research_drug='azithromycin'

import pandas as pd
import numpy as np
import functools
import re
import os
import json
from pprint import pprint
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import seaborn as sns
!pip install --upgrade "ibm-watson>=4.4.0"

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
def extract_research_drug(text,word):
    text=text.replace('covid-19','')
    text=text.lower()
    extract=''
    res=''
    try:
        re.search(word, text).group(0)
    except:
        # not found, return extract=null
        return extract
    
    res = [i.start() for i in re.finditer(word, text)]
    for result in res:
        extracted=text[result:result+200]
        extract=extract+' '+extracted
        if extract==' ':
            extract=''
    return extract  
    


def extract_conclusion(text,word):
    text=text.replace('covid-19','')
    text=text.lower()
    extract=''
    res=''
    try:
        re.search(word, text).group(0)
        res = [i.start() for i in re.finditer(word, text)]
        for result in res:
            extracted=text[result:result+200]
            extract=extract+' '+extracted
    except:
        # not found, return extract=null
        return extract
    return extract  
    
    
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
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
print ('ALL CORD19 articles',df.shape)

#fill na fields
df=df.fillna('no data provided')

#clean the abstract column so the filler 'do data provided'doesn't mess up with the real text later
df.abstract[df.abstract == 'no data provided'] = '' 

#drop duplicate titles
df = df.drop_duplicates(subset='title', keep="first")
#keep only 2020 dated papers
df=df[df['publish_time'].str.contains('2020')]
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()
df=search_focus(df)
print ('Keep only COVID-19 related articles',df.shape)
#define new column to host a snippet from the research paper related to the selected drug
df['paper'] = ''


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

#let's read the research papers 
for index, row in df.iterrows():
    file_fullpath = '/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files']
    if ';' not in row['sha'] and os.path.exists(file_fullpath)==True:
        with open(file_fullpath) as json_file:
            data=json.load(json_file)
            body=format_body(data['body_text'])
            body=body.replace("\n", " ")
            text=row['abstract']+' '+body.lower()
         
            #paper = research drug related text extracted from the research paper 
            df.loc[index, 'paper'] = extract_research_drug(json.dumps(data['body_text']).lower(), research_drug)
            
#drop columns we don't need anymore    
df=df.drop(['pdf_json_files'], axis=1)
df=df.drop(['sha'], axis=1)
df.head()
focus=research_drug
df1 = df[df['paper'].str.contains(focus)]


#if paper column is empty, let's replace it with the abstract
#remove lines without papers 
df1['paper'].dropna(how='any',inplace=True)

print(focus,'focused articles',df1.shape)

df1.head()

def extract_design(text,word):
    text=text.replace('covid-19','')
    extract=''
    res=''
    if word in text:
        res = [i.start() for i in re.finditer(word, text)]
    if res=='':
        word='study'
        res = [i.start() for i in re.finditer(word,text)]
    for result in res:
        extracted=text[result-400:result+400]
        #print (extracted1)
        extract=extract+' '+extracted
    #print (extract)
    return extract



#Create the Results Dataframe, sorted by date
df_results = pd.DataFrame(columns=['date','study','link','abstract','paper','source_x','journal','cord_uid'])

for index, row in df1.iterrows():
    abstract=df1.loc[index, 'abstract']
    paper=df1.loc[index, 'paper']
    #add link
    link=row['doi']
    linka='https://doi.org/'+link

    to_append = [row['publish_time'],row['title'],linka,abstract,paper,row['source_x'],row['journal'],row['cord_uid']]
    df_length = len(df_results)
    df_results.loc[df_length] = to_append
    

df_results=df_results.sort_values(by=['date'], ascending=False)
df_results.head()
#You can get one for free at https://www.ibm.com/watson/natural-language-processing
apikey=''
url=''
#Authenticate
authenticator = IAMAuthenticator(apikey)
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    authenticator=authenticator
)

natural_language_understanding.set_service_url(url)

#create two new columns - score and sentiment, which will have the values returned from the AI
df_results['score'] = np.nan
df_results['sentiment'] = np.nan

#reset index to zero
df_results.reset_index(drop=True, inplace=True)


for index, row in df_results.iterrows():
    try:
        #call the API
        response = natural_language_understanding.analyze(
            text=row['paper'],
            features=Features(keywords=KeywordsOptions(sentiment=True,emotion=False,limit=100))).get_result()
    except:
        #error, probably unsupported language, jump to the next one
        next
        
    for i in response['keywords']:
        if research_drug in i['text']:
            #found it, set the values and skip to the next row
            score = i['sentiment']['score']
            sentiment = i['sentiment']['label']
            df_results['score'][index] = score
            df_results['sentiment'][index] = sentiment
            break

df_results.head()
df_results.to_csv(research_drug+'_sentiment.csv',index=False)
#remove lines where the AI couldn't capture a score
df_results['score'].dropna(inplace=True)


grouped = df_results.groupby(df_results['sentiment'])

print('Total: ' + str(df_results['score'].size))
grouped['study'].count()
sns.set(style="darkgrid")
sns.set(font_scale=1.2)
sns.catplot(x='sentiment', data=df_results, kind="count", height=6, aspect=1.0, palette="hls")
plt.title('Sentiment Analysis for ' + research_drug)
plt.show();
