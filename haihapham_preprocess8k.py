# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import spacy 

from spacy.lang.en import English

from spacy import displacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English



import en_core_web_sm

import en_core_web_lg

import string



from numpy import *

from datetime import datetime

#nlp = en_core_web_sm.load()

nlp = en_core_web_lg.load()

#nlp = spacy.load("en")
nlp.max_length = 3e9
filename = "../input/AAPL-8K-20031015162950.txt"

document = open(filename).read()

doc_test = nlp(document)
def extract_content(orig_file_name):

    doc = open(orig_file_name).read()

    components = doc.split('TIME:')

    FileName =  components[0]

    StockCode = FileName.split('/')[0]

    StockCode = StockCode.split(':')[1]



    other = components[1]

    # DateTime after before 'EVENT:'

    components = other.split('EVENTS:')

    DateTime = components[0]

    DateTime = DateTime.replace('\n','')

    #DateTime

    Date = DateTime[0:8]

    Date = Date[0:4]+'-'+Date[4:6]+'-'+Date[6::]

    #Time = DateTime[8::]

    #Time = Time[0:2]+'-'+Time[2:4]+'-'+Time[4::]

    # Categories before "TEXT:", content after 'TEXT:'

    components = components[1]

    components = components.split('TEXT:')

    categories = components[0]

    categories = categories.split('\t')[1:]

    categories[-1] = categories[-1].replace('\n','')

    categories = [element.upper() for element in categories]

    #if '' in categories:

    #    categories = categories.remove('')

    

    content = components[1]

    return StockCode, Date, categories, content

#example

StockCode, Date, categories, content = extract_content(filename)



punctuations = string.punctuation

stopwords = spacy.lang.en.STOP_WORDS



def clean_component(doc):

    """ Clean up text. Tokenize, lowercase, and remove punctuation and stopwords """

    #print("Running cleaner")

    #Replace punctuation by spacing

    doc= doc.replace(punctuations, ' ')

    #remove multiple inline

    doc = doc.replace('\n',' ')

    doc = doc.strip()

    #Remove multiple space

    #doc = doc.strip()

    # Remove symbols (#) and stopwords

    doc = nlp(doc)

    doc = [tok.text for tok in doc if (tok.text not in stopwords and tok.pos_ != "PUNCT" and tok.pos_ != "SYM")]

    # Make all tokens lowercase

    doc = [tok.lower() for tok in doc]

    doc = ' '.join(doc)

    return nlp.make_doc(doc)



def pipe_clean(docs, **kwargs):

    for doc in docs:

        yield clean_component(doc)



# Yes, adding attributes to functions works...It's just a bit dirty-looking. Arguably less confusing to

# make it a class. Shrug.

clean_component.pipe = pipe_clean
Date_form = 'Date'





def date_index(data):

    data['datetime'] = pd.to_datetime(data[Date_form])

    data = data.set_index('datetime')

    data.drop([Date_form], axis=1, inplace=True)

    return data
def create_data(orig_file_name):

    #extract partial information from full report

    StockCode, datetime, categories, content = extract_content(orig_file_name)

    #if '' in categories:

    #    categories = categories.remove('')

    #clean content of report

    content_cleaned = clean_component(content)

    #vectorize content

    content_vector = content_cleaned.vector

    #create data frame

    content_vector = pd.DataFrame(content_vector).transpose()

    data= pd.DataFrame([1] * len(categories)).transpose()

    data.columns = categories

    data.insert(0, Date_form, datetime) 

    data.insert(1, "StockCode", StockCode)   

    

    data = pd.concat([data, content_vector], axis=1, sort=False)

   

    return data
list_file = os.listdir("../input")



def concat_report_data(list_file):

    Data = pd.DataFrame() 

    for file_name in list_file:

        orig_file_name = '../input/'+ file_name

        data = create_data(orig_file_name)

        Data = pd.concat([Data, data], axis=0, ignore_index=True, sort=False)

    #fill all NaN by 0, corresponding to the case that the categorie with NaN is not in the report

    Data = Data.fillna(0)

    Data[Date_form] = pd.to_datetime(Data[Date_form])

    Data = date_index(Data)

    return Data

#example

concat_report_data(list_file)
