# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer , WordNetLemmatizer
ps = PorterStemmer()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
example_sent = "Google LLC[5] is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, search engine, cloud computing, software, and hardware. Google was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University, California. Together, they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a privately held company on September 4, 1998. An Initial public offering (IPO) took place on August 19, 2004, and Google moved to its new headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google, Alphabet's leading subsidiary, will continue to be the umbrella company for Alphabet's Internet interests. Upon completion of the restructure, Sundar Pichai was appointed CEO of Google, replacing Larry Page, who became the CEO of Alphabet."
 
stop_words = set(stopwords.words('english'))
 
word_tokens = word_tokenize(example_sent)

#Punctuation removal
new_word_list12 = []
for w in word_tokens:
    
    arg2 = w not in string.punctuation
    value = arg2
    if(w not in string.punctuation):
        new_word_list12.append(w)

#integer value removal
new_list22 = []
for word in new_word_list12:
    try:
        value = float(word)
        continue
    except:
        new_list22.append(word)

#Stemming of data
new_list3 = []
for w in new_list22:
    new_list3.append(ps.stem(w))
print(new_list3)

#Stopword Removal
new_list33 = []
for w in new_list22:
    if w not in stop_words:
        new_list33.append(w)
print(new_list33)
