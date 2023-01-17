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
f=open("/kaggle/input/covid-dialogue-dataset/COVID-Dialogue-Dataset-English.txt","r")

a=f.read()
t=open("/kaggle/input/covid-dialogue-dataset/COVID-Dialogue-Dataset-Chinese.txt","r")

z=t.read()
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
stop_words=set(stopwords.words('english'))

word_tokens=word_tokenize(a)

filtered_sentence=[id for id in word_tokens if not id in stop_words]

filtererd_sentence=[]

for id in word_tokens:

    if id not in stop_words:

        filtered_sentence.append(id)

       

print(word_tokens)

print(filtered_sentence)
from wordcloud import WordCloud,STOPWORDS
stopwords=set(STOPWORDS)

stopwords.add("Dialogue Patient")

word_cloud=WordCloud(background_color="white",stopwords=stopwords).generate(a)
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(8,5))

plt.imshow(word_cloud)

plt.show()