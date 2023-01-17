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
!pip install tweet-preprocessor
!pip install emoji
!pip install textblob
import pandas as pd
import numpy as np
import preprocessor as p
import emoji
import re
comm = pd.read_csv('../input/youtube/GBcomments.csv',error_bad_lines=False,nrows=300)

comm.head()#just to display cells 
def extract_emojis(s):
  return (''.join(c for c in s if c in emoji.UNICODE_EMOJI))
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer

set(stopwords.words('english'))


stop_words = set(stopwords.words('english')) 
def stemfunction(text):  
    word_tokens = word_tokenize(text) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 

    Stem_words = []
    ps =PorterStemmer()
    for w in filtered_sentence:
        rootWord=ps.stem(w)
        Stem_words.append(rootWord)
    return(Stem_words)

from textblob import Word
def lemmitizer(stemmer_list):
    after_lem=[]
    for j in stemmer_list:
        p=Word(j)
        q=p.lemmatize()
        after_lem.append(q)
    return after_lem
    
from textblob import TextBlob
aa=[]
emoji_gb=[]
stem_gb=[]
lemmi_gb=[]

for i in comm.comment_text:
    em=extract_emojis(str(i))
    emoji_gb.append(em)
    i=p.clean(str(i))
    i =i.rstrip("\n")  
    i = ' '.join(re.sub("[\.\,\!\?\:\;\-\=\$\'\"\,\*\&\^\@]", " ",i).split())
    i = ' '.join(re.sub("(\w+:\/\/\S+)", " ", i).split())
    i=i.lower()
    # i=TextBlob(i)
    #i=str(i)
    st=stemfunction(i)
    stem_gb.append(st)
    lemmi=lemmitizer(st)
    lemmi_gb.append(lemmi)
    aa.append(i)
pol=[]

pos=[]
for cha in aa:
    bloby = TextBlob(cha)
    polar = bloby.sentiment.polarity
    pol.append(polar)
    posy = TextBlob(cha)
    flirt = posy.tags
    pos.append(flirt)
    
comm.insert(3,"processed_comments",aa)
comm.insert(4,"emojis_seperated",emoji_gb)
comm.insert(5,"stem_words",stem_gb)
comm.insert(6,"lemmitize_words",lemmi_gb)
comm.insert(7,"polarity of words",pol)
comm.insert(8,"parts of speach",pos)
comm.head(50)
comm.to_csv('file_gb.csv',index=False) 
