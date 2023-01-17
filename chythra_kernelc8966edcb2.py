
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import re
import string
import os
print(os.listdir("../input"))
#my_data = pd.read_csv('../input/train.csv')
faq = pd.read_csv('../input/500_questions.csv',encoding = "ISO-8859-1")
topic=pd.read_csv('../input/topic_dataset.csv',encoding = "ISO-8859-1")
#print(faq)
#removing punctions,numbers,text inside brackets etc
def remove_punctuation(x):
    # Removing non ASCII chars
    x = re.sub("[^\x00-\x7f]", " ",str(x))
    x =re.sub(r'\(.*\)', '', str(x))
    #x=x.replace(r"\(.*\)","")
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)

faq['question'] = faq['question'].apply(remove_punctuation)
faq.question = faq.question.str.replace('\d+\s', '')   
#faq['question'] = faq['question'].map(stripTags)
#faq.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
#topic.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
#faq.columns = map(str.lower, faq.columns)
faq =faq.astype(str).apply(lambda x: x.str.lower())

topic_list = topic['topic'].tolist()


for topics in topic_list:
    #faq['topic'] = faq.astype(str).sum(axis=1).str.contains(topic)
    faq.loc[faq['question'].str.contains(topics),'topic'] = topics
 
faq = faq[pd.notnull(faq['topic'])]

faq=faq.merge(topic, on="topic", how = 'inner')
print(faq)
faq=faq.groupby(['subject','topic'])
print(faq)
# Any results you write to the current directory are saved as output.

