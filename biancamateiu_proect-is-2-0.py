# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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



df = {"paper_id": [], "section_id": [], "section_body": [], "tag_label" : []}

df = pd.DataFrame.from_dict(df)

df
keywords = ['2019-ncov', '2019 novel coronavirus', 'coronavirus 2019', 'coronavirus disease 19', 'covid-19', 'covid 19', 'ncov-2019', 'sars-cov-2', 'wuhan coronavirus', 'wuhan pneumonia', 'wuhan virus']



#tag sections that may be realted to coronavirus



def generate_label_tag(article):

    tags = None

    if any(x in article.lower() for x in keywords):

        tags = "COVID-19"

    return tags
def create_row(text, section_id, section_label, paper_id):

    row = {"paper_id": None, "section_id": None, "section_body": None, "grammar_label": None, "tag_label" : None}

    row['paper_id'] = paper_id

    row['section_id'] = paper_id + '_' + str(section_id)

    row['section_body'] = text

    row['grammar_label'] = None

    row['tag_label'] = section_label

    

    return row
stopwords = ['copyright', 'preprint', 'permission', 'rights']



def stop_sentence(text):

    if any(x in text.lower() for x in stopwords):

        return True

    return False
collect_json = glob.glob(f'{root_path}/**/*.json', recursive=True) #finds all the pathnames matching a specified pattern





for i,file_name in enumerate (collect_json):

    if i%1000==0:

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

        

        if article_tag is None:

            continue

        

        section_id = 0

        paper_id = None

        try:

            paper_id = data['paper_id']

        except:

            pass

        

        try:

            for _ in range(len(data['abstract'])):

                text = data['abstract'][_]['text']

                if not stop_sentence(text):

                    row = create_row(text, section_id, article_tag, paper_id)

                    section_id = section_id + 1;

                    data_parsed = data_parsed.append(row, ignore_index=True)

        except:

            pass

        

        

        for i in range(len(body_list)):

            text = body_list[i]

            paragraph_sentence_list = []

            for x in sent_tokenize(text):

                if not stop_sentence(x):

                    paragraph_sentence_list.append(x)

                    

            paragraph = " ".join(paragraph_sentence_list)

            

            row = create_row(paragraph, section_id, article_tag, paper_id)

            section_id = section_id + 1;

            df = df.append(row, ignore_index=True)
df.shape
df.head()
df1 = df.describe(include = 'all')



df1.loc['dtype'] = df.dtypes

df1.loc['size'] = len(df)

df1.loc['% count'] = df.isnull().sum()



print (df1)
df.drop_duplicates(subset="section_body", inplace = True)
df.shape
analyze_data_dict = {"section_id" : [], "section_body": [], "length": []}

analyze_data = pd.DataFrame.from_dict(analyze_data_dict)



analyze_data['section_body'] = df['section_body']

analyze_data['section_id'] = df['section_id']



analyze_data['length'] = df['section_body'].apply(len)



analyze_data.sort_values(by = ['length'], inplace=True)



analyze_data.head()
analyze_data = analyze_data[analyze_data.length > 30]



analyze_data.shape
merge = pd.merge(df, analyze_data, on = ['section_id', 'section_body'])

merge.head()
merge.shape
merge.to_csv('mycsvfile.csv',index=False)