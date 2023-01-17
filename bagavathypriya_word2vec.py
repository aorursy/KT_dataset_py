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
paragraph="Many questions were arising in my mind, will she would be friends with a person like me?? Its not that I was coward or was scared to tell her about my feelings but there was one reason to my hesitance in speaking to her. I “stammer” while speaking, because of this reason only I had no friends .I didn’t want to talk to anybody because sooner or later they used to make fun of me, so I just decide to stay away from my classmates and other friends. I was shy to speak in front of the crowd, just wanted to stay secluded and lonely . But this time I was thinking something else, I really wanted that girl to be love of my life on any condition. But I was also aware of the fact that no girl would like that her boyfriend stammers, not even if I’m the only man left on earth."
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
sentences=nltk.sent_tokenize(paragraph)
sentences=[nltk.word_tokenize(sentence) for sentence in sentences]
for i in range(0,len(sentences)):
    sentences[i]=[word for word in sentences[i] if word not in stopwords.words('english')]
sentences
model=Word2Vec(sentences,min_count=1)
words=model.wv.vocab
words
vec=model.wv['girl']
vec
vec.shape
sim=model.wv.most_similar('friends')
sim
