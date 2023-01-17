# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
meta=pd.read_csv("/kaggle/input/inteligent-systems-project-covid-19/mycsvfile.csv")

meta.head()
meta.shape
is_covid = meta['tag_label'] == 'COVID-19'

data = meta[is_covid]
data.shape
data_parsed = {"paper_id": [], "section_id": [], "title" : [], "abstract" : [], "publish_time" : [], 

               "authors" : [], "url" : [], "section_body": [], "grammar_label": [], "tag_label" : []}

data_parsed = pd.DataFrame.from_dict(data_parsed)

data_parsed
import spacy

from nltk.tokenize import sent_tokenize

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

    

    
def create_row(row, text, section_id):

    new_row = {"paper_id": None, "section_id": None, "title" : None, "abstract" : None, "publish_time" : None, 

               "authors" : None, "url" : None, "section_body": None, "grammar_label": None, "tag_label" : None}

    new_row['paper_id'] = row['paper_id']

    new_row['section_id'] = row['paper_id'] + '_' + str(section_id)

    new_row['title'] = row['title']

    new_row['abstract'] = row['abstract']

    new_row['publish_time'] = row['publish_time']

    new_row['authors'] = row['authors']

    new_row['url'] = row['url']

    new_row['section_body'] = text

    new_row['grammar_label'] = generate_grammar_label(text)

    new_row['tag_label'] = row['tag_label']

    

    return new_row
count = 0

for i, row in data.iterrows():

    ##add title as a section

    if count%100==0:

        print ("====processed " + str(count)+ ' rows=====')

        print()

    count = count + 1

        

    section_id = 0

    try:

        new_row = create_row(row, row['title'], section_id)

        data_parsed = data_parsed.append(new_row, ignore_index=True)

    except:

        pass

        

    ##add abstract as sections

    try:

        for x in sent_tokenize(row['abstract']):

            section_id = section_id + 1;

            new_row = create_row(row, x, section_id)

            data_parsed = data_parsed.append(new_row, ignore_index=True)

    except:

        pass

    #add full text as sections

            

    try:

        for x in sent_tokenize(row['section_body']):

            section_id = section_id + 1;

            new_row = create_row(row, x, section_id)

            data_parsed = data_parsed.append(new_row, ignore_index=True)

    except:

        pass

data_parsed.shape
data_parsed.head()
is_fragment = data_parsed['grammar_label'] == 'FRAGMENT'

data_fragment = data_parsed[is_fragment]

data_fragment.shape
is_quest = data_parsed['grammar_label'] == 'QUESTION'

data_quest = data_parsed[is_quest]

data_quest.shape
data_parsed.to_csv('mycsvfile.csv',index=False)