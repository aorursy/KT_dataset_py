

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import gc
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Any results you write to the current directory are saved as output


root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})

meta_df.head()


all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
# see how many papers are there
len(all_json)
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            print(content.keys())
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            try :
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            except Exception as e:
                pass
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)

                
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)


def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data


dict_ = {'paper_id': [], 'abstract': [], 'body_text': [],'publish_year':[],'url':[],'WHO #Covidence':[] ,'license':[],'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
for idx, entry in enumerate(all_json[:10000]):
    if idx % (len(all_json) // 100) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    dict_['publish_year'].append(pd.DatetimeIndex(meta_data['publish_time']).year.values[0])   
    dict_['url'].append(meta_data['url'].values[0])
    dict_['WHO #Covidence'].append(meta_data['WHO #Covidence'].values[0])
    dict_['license'].append(meta_data['license'].values[0])
    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0: 
        # no abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 300 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract_summary'].append(summary)
        
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    
    try:
        # if more than one author
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            # more than 2 authors, may be problem when plotting, so take first 2 append with ...
            dict_['authors'].append(". ".join(authors[:2]) + "...")
        else:
            # authors will fit in plot
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if only one author - or Null valie
        dict_['authors'].append(meta_data['authors'].values[0])
    
    # add the title information, add breaks when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # if title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['title'].values[0])
    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'publish_year','url','license','WHO #Covidence','body_text', 'authors', 'title', 'journal', 'abstract_summary'])
df_covid.head()


df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
df_covid['abstract'].describe(include='all')


df_covid.dropna(inplace=True)
df_covid.info()
# import re
# df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
# df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))
df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))


from nltk.corpus import stopwords
stop = stopwords.words('english')

stop_words = ["fig","figure", "et", "al", "table",  
        "data", "analysis", "analyze", "study",  
        "method", "result", "conclusion", "author",  
        "find", "found", "show", "perform",  
        "demonstrate", "evaluate", "discuss", "google", "scholar",   
        "pubmed",  "web", "science", "crossref", "supplementary", "A" ,"this" ,"that",'the', 'et','al','in','also' ,'and']

stop_words = stop + stop_words
df_covid['body_text'] = df_covid['body_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
df_covid['body_text'].head()

covid_words = ['covid-19','sars-cov-2','2019-ncov','sars' ,'coronavirus' ,'novel coronavirus','corona','cov-19']
is_covid19_article = df_covid.body_text.str.contains('covid-19|sars-cov-2|2019-ncov|sars coronavirus 2|2019 novel coronavirus|COVID-19|SARS-CoV-2|2019-nCov|SARS Coronavirus 2|2019 Novel Coronavirus')
df_covid = df_covid[is_covid19_article]
df_covid
df_covid['body_text'] = df_covid['body_text'].apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))
df_covid['body_text'].head()
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process 
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

question = "What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?"

queries = "  Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.\
             Prevalence of asymptomatic shedding and transmission (e.g., particularly children). \
             Seasonality of transmission.\
             Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).\
             Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).\
             Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).\
             Natural history of the virus and shedding of it from an infected person. \
             Implementation of diagnostics and products to improve clinical processes. \
             Disease models, including animal models for infection, disease and transmission.\
             Immune response and immunity.\
             Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings. \
             Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings. \
             Role of the environment in transmission.\
             Tools and studies to monitor phenotypic change and potential adaptation of the virus."


#print(sent_tokenize(question))
stop_words += ['What', 'known', 'even', 'and']
queries = sent_tokenize(question + queries)
queries = pd.DataFrame(queries,columns=['queries'])
queries['queries'] = queries['queries'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
queries

import re
queries['queries'] = queries['queries'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
queries['queries'].apply(lambda x: lower_case(x))
queries
txt_list = df_covid['body_text'].tolist()
queries_list = queries['queries'].tolist()
print("Query is: ",queries_list[0])
res = process.extract(queries_list[0], sent_tokenize(df_covid['body_text'].values[1]))
res
def highlight_many(text, keywords1,keywords2 = ""):
    replacement = "\033[91m" + "\\1" + "\033[39m"
    replacement1 = "\033[94m" + "\\1" + "\033[39m"
    text = re.sub("(" + "|".join(map(re.escape, keywords1)) + ")", replacement, text, flags=re.I)
    text = re.sub("(" + "|".join(map(re.escape, keywords2)) + ")", replacement1, text, flags=re.I)
    print (text)
print(word_tokenize(queries['queries'].tolist()[0]))
highlight_many(res[0][0],word_tokenize(queries['queries'].tolist()[0]),covid_words)
highlight_many(df_covid['body_text'].values[1],word_tokenize(queries['queries'].tolist()[0]),covid_words)
query = queries_list[11]
print("Query is ", query,end='\n\n')
res = process.extract(query,txt_list)
print(highlight_many(res[0][0],word_tokenize(query),covid_words))