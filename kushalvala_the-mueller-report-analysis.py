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
import nltk

import string

import re

from wordcloud import WordCloud, STOPWORDS

data = pd.read_csv("/kaggle/input/mueller-report/mueller_report.csv")
data.head()
data.shape
data['text'] = data['text'].astype('str')
wn = nltk.WordNetLemmatizer()

stopword = nltk.corpus.stopwords.words('english')





def clean_text(text):

    text_lr  = "".join([char for char in text if char not in string.punctuation])

    text_rc = re.sub('[0-9]+', '', text_lr) # remove puntuation

    tokens = re.split('\W+', text_rc)    # tokenization

    text = [wn.lemmatize(word) for word in tokens if word not in stopword]  # remove stopwords and stemming

    return ' '.join(text)
data['cleaned_text'] = data['text'].apply(lambda x : clean_text(x))
data.head(10)
text_data = ' '.join(data['cleaned_text'].tolist())
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel2', collocations=False, stopwords = STOPWORDS).generate(text_data)
import matplotlib.pyplot as plt

plt.figure(figsize=(40, 30))

plt.imshow(wordcloud)

plt.axis("off")
from nltk.probability import FreqDist

words = nltk.tokenize.word_tokenize(text_data)

fdist = FreqDist(words)
plt.figure(figsize=(20,10))

fdist.plot(100, cumulative = False)
print('Number of words counted:' , fdist.N())