# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import sys

import json

import glob

from  collections import OrderedDict



import spacy

from nltk.tokenize import sent_tokenize



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# the data preprocessing was inspired by this Kernel https://www.kaggle.com/latong/extractive-text-summarization



#read the metadata file



meta=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

meta.head()
meta.shape
#create another dataFrame, containing only the 4 specified columns

#drop the rows that don't have an id, or that are dupicates



meta_important = meta[["sha", "title", "abstract", "publish_time", "authors", "url"]]

meta_important.columns = ["paper_id", "title", "abstract", "publish_time", "authors", "url"]

meta_important = meta_important[meta_important["paper_id"].notna()]

meta_important.drop_duplicates(subset="title", keep = False, inplace = True)

meta_important.head()
meta_important.shape
sys.path.insert(0, "../")



root_path = '/kaggle/input/CORD-19-research-challenge/'

#inspired by this kernel. Thanks to the developer ref. https://www.kaggle.com/fmitchell259/create-corona-csv-file

# Just set up a quick blank dataframe to hold all these medical papers. 



df = {"paper_id": [], "section_id": [], "section_body": [], "grammar_label": [], "tag_label" : []}

df = pd.DataFrame.from_dict(df)

df
keywords = ['2019-ncov', '2019 novel coronavirus', 'coronavirus 2019', 'coronavirus disease 19', 'covid-19', 'covid 19', 'ncov-2019', 'sars-cov-2', 'wuhan coronavirus', 'wuhan pneumonia', 'wuhan virus']



#tag sections that may be realted to coronavirus



def generate_label_tag(article):

    tags = None

    if any(x in article.lower() for x in keywords):

        tags = "COVID-19"

    return tags


spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])

"""

tokenize the text using spacy. We want to identinfy the non-informative sentences

we consider non-informative sentences to be questions, or fragments that do NOT

contain a noun AND a verb AND at least 5 words

"""

def generate_grammar_label(section):

    if section.strip().endswith("?"):

        return "QUESTION"

    

    tokens = spacy_nlp(section)

    for token in tokens:

        if token.pos == "NUM" and token.text.lower() == "2019-ncov":

            token.pos = "NOUN"

     

    #nouns

    nouns = any([t.pos_  in ["NOUN", "PROPN"]  and t.dep_ in ["nsubj", "nsubjpass"] for t in tokens])



    # Actions (adverb, auxiliary, verb)

    action = any([t.pos_ in ["ADV", "AUX", "VERB"] for t in tokens])



     # Consider root words with a nominal subject as an action word

    action = action or any([t.dep_ in ["appos", "ROOT"] and any([x.dep_ in ["nsubj", "nsubjpass"] for x in t.children]) for t in tokens])



    # Non-punctuation tokens and multi-character words (don't count single letters which are often variables used in equations)

    words = [t.text for t in tokens if t.pos_ not in ["PUNCT", "SPACE", "SYM"] and len(t.text) > 1]



    # Valid sentences take the following form:

    #  - At least one nominal subject noun/proper noun AND

    #  - At least one action/verb AND

    #  - At least 5 words

    valid = nouns and action and len(words) >= 5

    

    if not valid:

        return "FRAGMENT"

    

    return None

    

    
def create_row(text, section_id, section_label, paper_id):

    row = {"paper_id": None, "section_id": None, "section_body": None, "grammar_label": None, "tag_label" : None}

    row['paper_id'] = paper_id

    row['section_id'] = paper_id + '_' + str(section_id)

    row['section_body'] = text

    row['grammar_label'] = None

    #row['grammar_label'] = generate_grammar_label(text)

    row['tag_label'] = section_label

    

    return row

    
#we are iterting over all json files containing data, and populating the dataframe created earlier



collect_json = glob.glob(f'{root_path}/**/*.json', recursive=True) #finds all the pathnames matching a specified pattern





for i,file_name in enumerate (collect_json):

    if i%2000==0:

        print ("====processed " + str(i)+ ' json files=====')

        print()



    with open(file_name) as json_data:

            

        data = json.load(json_data,object_pairs_hook=OrderedDict)

        

        body_list = []

       

        for _ in range(len(data['body_text'])):

            try:

                body_list.append(data['body_text'][_]['text'])

            except:

                pass



        body = "\n ".join(body_list)

        

        article_tag = generate_label_tag(body)

        paper_id = None

        try:

            paper_id = data['paper_id']

        except:

            pass

        

        row = create_row(body, None, article_tag, paper_id)

        df = df.append(row, ignore_index=True)



        ##add title as a section

        

#         section_id = 0

#         try:

#             row = create_row(data['metadata']['title'], section_id, article_tag, paper_id)

#             df = df.append(row, ignore_index=True)

#         except:

#             pass

        

#         ##add abstract as sections

        

#         for _ in range(len(data['abstract'])):

#             try:

#                 for x in sent_tokenize(data['abstract'][_]['text']):

#                     section_id = section_id + 1;

#                     row = create_row(x, section_id, article_tag, paper_id)

#                     df = df.append(row, ignore_index=True)

#             except:

#                 pass

            

        ##add full text as sections

            

#         for _ in range(len(data['body_text'])):

#             try:

#                 for x in sent_tokenize(data['body_text'][_]['text']):

#                     section_id = section_id + 1;

#                     row = create_row(x, section_id, article_tag, paper_id)

#                     df = df.append(row, ignore_index=True)

#             except:

#                 pass

            

    
df.head()
df.shape
merge = pd.merge(meta_important, df, on = ['paper_id'])

merge.head()


seriesObj = df.apply(lambda x: True if x['grammar_label'] == 'QUESTION' else False , axis=1)

numOfRows = len(seriesObj[seriesObj == True].index)

print('questions: ', numOfRows)
seriesObj = df.apply(lambda x: True if x['grammar_label'] == 'FRAGMENT' else False , axis=1)

numOfRows = len(seriesObj[seriesObj == True].index)

print('fragment: ', numOfRows)
seriesObj = df.apply(lambda x: True if x['tag_label'] == 'COVID-19' else False , axis=1)

numOfRows = len(seriesObj[seriesObj == True].index)

print('covid tag: ', numOfRows)
merge.to_csv('mycsvfile.csv',index=False)