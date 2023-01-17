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
train_data = pd.read_csv('/kaggle/input/ag-news-classification-dataset/train.csv')
test_data = pd.read_csv('/kaggle/input/ag-news-classification-dataset/test.csv')
train_data.head()
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.stem import PorterStemmer
import numpy as np
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,5))
labels = 'world', 'sports', 'business' , 'Science'
sizes = [len(train_data[train_data['Class Index']==1]), len(train_data[train_data['Class Index']==2]), len(train_data[train_data['Class Index']==3]), len(train_data[train_data['Class Index']==4])]
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')  
plt.show()
def tokenize(data):
    data = [word_tokenize(sent) for sent in data]
    return data
def remove_punc(data):
    for i in range(len(data)-1):
        data[i] = [w for w in data[i] if w not in string.punctuation]
    return data
nltk_stop_words = nltk.corpus.stopwords.words('english')
def remove_stop(data):    
    for i in range(len(data)-1):
        data[i] = [w for w in data[i] if w not in nltk_stop_words]
    return data
stemmer = PorterStemmer()
def stemming(data):    
    for i in range(len(data)-1):
        data[i] = [stemmer.stem(w) for w in data[i]]
    return data
lemmatizer = WordNetLemmatizer()
def lemma(data):    
    for i in range(len(data)-1):
        data[i] = [lemmatizer.lemmatize(w) for w in data[i]]
    return data
regex = re.compile('[^a-zA-Z]')
def remove_nonalpha(data):
    for i in range(len(data)-1):
        data[i] = [regex.sub('', w) for w in data[i]]
    return data
train_data['Title'] = tokenize(train_data['Title'])
train_data['Title'] = remove_punc(train_data['Title'])
train_data['Title'] = remove_stop(train_data['Title'])
train_data['Title'] = stemming(train_data['Title'])
train_data['Title'] = lemma(train_data['Title'])
train_data['Title'] = remove_nonalpha(train_data['Title'])
train_data['Description'] = tokenize(train_data['Description'])
train_data['Description'] = remove_punc(train_data['Description'])
train_data['Description'] = remove_stop(train_data['Description'])
train_data['Description'] = stemming(train_data['Description'])
train_data['Description'] = lemma(train_data['Description'])
train_data['Description'] = remove_nonalpha(train_data['Description'])
train_data.head()
