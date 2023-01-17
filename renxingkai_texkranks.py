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
file_related="/kaggle/input/cut/n1000_prf1.csv"

file_unrelated="/kaggle/input/cut/n288_prf1.csv"
data_related=pd.read_csv(file_related)

data_unrelated=pd.read_csv(file_unrelated)
data_related.head()
import jieba.analyse

import jieba
def textrank(text,topK=20):

    kw_textrank=jieba.analyse.textrank(text,topK=topK)

    return kw_textrank
def tfidf(text,topK=20):

    kw_tfidf=jieba.analyse.extract_tags(text,topK=topK)

    return kw_tfidf
from tqdm import tqdm
tqdm.pandas("my bar!")
data_unrelated['textrank']=data_unrelated['name_content_clean_cut'].progress_apply(textrank)
data_unrelated['tfidf']=data_unrelated['name_content_clean_cut'].progress_apply(tfidf)
data_related.fillna("我",inplace=True)
data_related['textrank']=data_related['name_content_clean_cut'].progress_apply(textrank)
data_related['tfidf']=data_related['name_content_clean_cut'].progress_apply(tfidf)
data_related.head()
textrank_list=[]

for i in tqdm(range(len(data_unrelated))):

    for j in range(len(data_unrelated['textrank'][i])):

        textrank_list.append(data_unrelated['textrank'][i][j])
data_related.to_csv("textrank_tfidf.csv",index=False,encoding='utf-8')
data_unrelated.to_csv("n288_textrank_tfidf.csv",index=False,encoding='utf-8')