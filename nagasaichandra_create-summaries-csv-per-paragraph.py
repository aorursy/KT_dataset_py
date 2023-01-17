# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv

df = pd.read_csv('/kaggle/input/stanford-plato-corpus/data_per_paragraph.csv')

unique_titles_df = pd.read_csv('/kaggle/input/stanford-plato-corpus/corpus_articles.csv')



sample_10 = unique_titles_df[['filename', 'title']].sample(n=10)
sample_10
x_df = pd.read_csv('/kaggle/input/stanford-plato-corpus/sample_summaries_per_document.csv')
x_df.shape
import transformers

from transformers import pipeline

import re

summarizer = pipeline('summarization')





def _with_counter(fun, _N, _i=1):

    """

    print a progress count with each function call

    """

    global COUNTER_I, COUNTER_N

    # initialize for reuse within _fun

    COUNTER_I = _i

    COUNTER_N = _N

    def _fun(*args, **kwargs):

        global COUNTER_I, COUNTER_N

        print(

            f"{COUNTER_I/COUNTER_N}\t == \t{COUNTER_I} / {COUNTER_N}", 

            end="\r"

        )

        print("DONE",f"{COUNTER_I} / {COUNTER_N}","\t"*30) if COUNTER_I >= COUNTER_N else None

        COUNTER_I += 1

        return fun(*args, **kwargs)

    return _fun





def summarize(text, max_length=512, min_length=100):

    key='summary_text'

    summary_list = summarizer(text, max_length=max_length, min_length=min_length)

    summary = summary_list[0][key]

    return summary



# summaries = paragraphs.apply(csummarize)

def simple_preprocess(text):

    text = text.replace('\n', ' ') #Remove line breaks

    matches = re.findall(r'\(([^()]*)\)', text)  #Remove citations

    for i in matches:

        if re.search(r'[0-9][0-9][0-9][0-9]?', i):

            text = text.replace('({})'.format(i), '')

        elif len(i) < 20:

            text = text.replace('({})'.format(i), '')

    return text





#usage get_data(title)

#returns list of dictionaries, preamble

#where each dictionary has only one key and value, key is a tuple like (section_id, paragraph_id) and value is the paragraph_text. 



def get_data(title):

    df['preamble_text'] = df['preamble_text'].fillna(str(''))

    df['section.paragraph.text'] = df['section.paragraph.text'].fillna(str(''))

    if title in list(df['title']):

        req_df = df.loc[df['title'] == title]

        req_df['section.processed_paragraph.text'] = req_df['section.paragraph.text'].apply(simple_preprocess)

        csummarize = _with_counter(summarize, _N=len(req_df))



        req_df['summary'] = req_df['section.processed_paragraph.text'].apply(csummarize)

        data = [{(s,p_id):str(p)} 

                            for s,p_id,p in zip(req_df['section.id'], 

                                                req_df['section.paragraph.id'],

                                                req_df['summary'])]

#         data = ''

        preamble = list(req_df['preamble_text'])[0]

#         print(preamble)

#         if preamble is None:

#             return ''

#         else:

        preamble = simple_preprocess(preamble)

        return data, preamble

    
# x_df = pd.DataFrame(columns=['title', 'filename', 'preamble', 'data'])
x_df.head()
data_list = list()

preamble_list = list()

title_list = list()

filename_list = list()

# with open('/sample_summaries_per_document.csv', 'a') as f:

# writer = csv.writer(f)

# writer.writerow(['title', 'filename', 'preamble', 'data'])



for i in range(len(list(sample_10))):

    print('Processing article: {}/9'.format(i))

    data, preamble = get_data(list(sample_10['title'])[i])

    data_list.append(data)

    filename_list.append(list(sample_10['filename'])[i])

    title_list.append(list(sample_10['title'])[i])

    preamble_list.append(preamble)

    x_df = x_df.append({'title':list(sample_10['title'])[i], 'filename':list(sample_10['filename'])[i], 'preamble':preamble, 'data':data}, ignore_index=True)

#         writer = csv.writer(f)

#         writer.writerow([list(sample_10['title'])[i], list(sample_10['filename'])[i], preamble, data])

    
x_df.to_csv('sample_summaries_per_document.csv')
pd.read_csv('sample_summaries_per_document.csv').head()
print(filename_list)

print(title_list)
!ls

# data, preamble = get_data(title)xds

# data


# preamble



