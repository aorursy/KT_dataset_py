# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install tokenizers
!pip install transformers
from __future__ import print_function

import ipywidgets as widgets

from transformers import pipeline
import glob

txt_files = glob.glob("/kaggle/input/toi-2018-news-articles/data/*.txt")

txt_files[0:10]
from collections import defaultdict

count = len(txt_files)

data = defaultdict(list)

for i in txt_files:

    date = i.split('/')[-1].split('_')[0]

    topic = '_'.join(i.split('/')[-1].split('_')[1:-1])

    headline = i.split('/')[-1].split('_')[-1][0:-4]

    

    with open(i, 'rt') as fd:

        data[date].append([topic,headline,fd.read().strip('[]')])

    count-=1

    if(count%10000==0):

        print('{} files are left to be processed...'.format(count))

article = data['201824'][0][2]

print(article)
nlp_sentence_classif = pipeline('sentiment-analysis')

nlp_sentence_classif(','.join(article))
nlp_qa = pipeline('question-answering')

nlp_qa(context=article , question='What does the police think about the wherebaouts of the thieves ?')