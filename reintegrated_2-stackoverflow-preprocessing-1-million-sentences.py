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
import pandas as pd

data=pd.read_csv("../input/stackoverflow-tag-prediction/data.csv")

data.drop(columns=['Unnamed: 0','Id'],axis=1,inplace=True)

data
def strip_html(sentence):

    compiled=re.compile(r"<.*?>")

    cleaned_sentence=re.sub(compiled," ",str(sentence))

    return cleaned_sentence

import re

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

sentences=[]

stemmer=SnowballStemmer("english")

stop_words = set(stopwords.words('english'))

start=0

end=1000000

from datetime import datetime

start_time = datetime.now()

for i,j in zip(data.iloc[start:end,0],data.iloc[start:end,1]):

    sentence=str(i)+" "+str(j)

    sentence=re.sub(r"<code>(.*?)</code>"," ",sentence)# removes all the code

    sentence=strip_html(sentence.encode('utf-8'))

    sentence=re.sub(r"[^A-Za-z]"," ",sentence)

    words=word_tokenize(sentence.lower())

    sentence=" ".join(str(stemmer.stem(i)) for i in words if i not in stop_words and( len(i)!=1 or i=='c'))

    sentences.append(sentence)

print(datetime.now()-start_time)

import pickle as pkl

with open('data1.txt','wb') as fh:

    pkl.dump(sentences,fh)