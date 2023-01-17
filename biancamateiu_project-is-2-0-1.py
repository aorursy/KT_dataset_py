# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import tensorflow_hub as hub



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/proect-is-2-0/mycsvfile.csv")

data.head()
data.shape
data_parsed = {"paper_id": [], "section_id": [], "section_body": [], "grammar_label": [], "tag_label" : []}

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

    

    return "INFORMATION"
def create_row(row, grammar):

    new_row = {"paper_id": None, "section_id": None, "section_body": None, "grammar_label": None, "tag_label" : None}

    new_row['paper_id'] = row['paper_id']

    new_row['section_id'] = row['section_id']

    new_row['section_body'] = row['section_body']

    new_row['grammar_label'] = grammar

    new_row['tag_label'] = row['tag_label']

    

    return new_row
for i, row in data.iterrows():

    ##add title as a section

    if i%10000==0:

        print ("====processed " + str(i)+ ' rows=====')

        print()

        

    text = row['section_body']

    

    is_info = True

    count = 0

    non_info = 0

    

    for x in sent_tokenize(text):

        count = count + 1

        label_sent = generate_grammar_label(x)

        if label_sent != 'INFORMATION':

            non_info = non_info + 1

        

    label = 'INFORMATION'

        

    if non_info > (count / 2):

        label = 'GIBBERISH'

        

    new_row = create_row(row, label)

    data_parsed = data_parsed.append(new_row, ignore_index=True)
data_parsed.head()
data_parsed.shape
is_fragment = data_parsed['grammar_label'] == 'GIBBERISH'

data_fragment = data_parsed[is_fragment]

data_fragment.shape
data_parsed.to_csv('mycsvfile.csv',index=False)