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
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
train
import re



alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

god_list = ['buddha', 'allah', 'jesus']

train_text_list = list(train['text'])

text_list_corpus = ''



for text in train_text_list:

    text_list_corpus = text_list_corpus + text



text_list_corpus  = text_list_corpus.lower()

text_list_corpus

pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+/(?:[-\w.]|(?:%[\da-fA-F]{2}))+' 

clear_text = re.sub(pattern=pattern, repl='LINK', string=text_list_corpus)

clear_text = clear_text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》;]', '', clear_text)

clear_text = clear_text = re.sub('[0-9]', 'num ', clear_text)

for i in range(len(alphabets)):

    clear_text = re.sub(alphabets[i]+'{3,}', alphabets[i], clear_text)

for i in range(len(god_list)):

    clear_text = clear_text.replace(god_list[i], 'god')

clear_text
import nltk



from nltk.corpus import stopwords  

stop_words = stopwords.words('english')



from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
word_list = word_tokenize(clear_text)

word_list = [stemmer.stem(word) for word in word_list]

word_list = [word for word in word_list if word not in stop_words]

len(list(set(word_list)))
from collections import Counter

count = Counter(word_list)

common_tag_200 = count.most_common(16208)

common_tag_200[:10]
from wordcloud import WordCloud

import matplotlib.pyplot as plt



wc = WordCloud(background_color="white", width=3200, height=2400)

cloud = wc.generate_from_frequencies(dict(common_tag_200))

plt.figure(figsize=(20, 16))

plt.axis('off')

plt.imshow(cloud)

plt.show()