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
df = pd.read_csv('../input/djinni-requirements.csv')

df.head()
df.loc[2, 'requirements']
# Importing libraries 

import nltk 

import re 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 



txt1 = df['requirements']

  

# Preprocessing 

def remove_string_special_characters(s): 

      

    # removes special characters with ' ' 

    stripped = re.sub('[^a-zA-z\s]', '', s) 

    stripped = re.sub('[0-9]+', '', stripped)

    stripped = re.sub('_', '', stripped) 

      

    # Change any white space to one space 

    stripped = re.sub('\s+', ' ', stripped) 

      

    # Remove start and end white spaces 

    stripped = stripped.strip() 

    if stripped != '': 

            return stripped.lower() 

          

# Stopword removal  

stop_words = set(stopwords.words('english')) 

your_list = ['skills', 'ability', 'job', 'description'] 

for i, line in enumerate(txt1): 

    txt1[i] = ' '.join([x for 

        x in nltk.word_tokenize(line) if 

        ( x not in stop_words ) and ( x not in your_list ) and ( not x.isdigit() )]) 

      

# Getting trigrams  

vectorizer = CountVectorizer(ngram_range = (3,3)) 

X1 = vectorizer.fit_transform(txt1)  

features = (vectorizer.get_feature_names()) 

print("\n\nFeatures : \n", features[:20]) 
# Applying TFIDF 

vectorizer = TfidfVectorizer(ngram_range = (3,3)) 

X2 = vectorizer.fit_transform(txt1) 

scores = (X2.toarray()) 

# print("\n\nScores : \n", scores) 

  

# Getting top ranking features 

sums = X2.sum(axis = 0) 

data1 = [] 

for col, term in enumerate(features): 

    data1.append( (term, sums[0,col] )) 

ranking = pd.DataFrame(data1, columns = ['term','rank']) 

words = (ranking.sort_values('rank', ascending = False)) 

words.head(10)
words.head(50)
df = pd.read_csv('../input/dou-req2.csv')

df.head()
df['Java'].value_counts()
df['Is junior'] = np.array([int('junior' in x.lower() or 'trainee' in x.lower() or 'intern' in x.lower()) 

                            for x in df['title']])
df['Is junior'].sum() / df.shape[0]
df['Is junior'].sum()
df.shape
df['Is senior'] = np.array([int('senior' in x.lower()) for x in df['title']])

df['Is senior'].sum()
df['Not senior'] = np.array([int('senior' not in x.lower() and

                                 'middle' not in x.lower() and

                                 'team lead' not in x.lower()) 

                             for x in df['title']])

df['Not senior'].sum()
df[ df['Not senior']==1 ].shape
df[ df['Not senior']==1 ]['Java'].value_counts()
txt1 = df[ (df['Not senior']==1) & (df['Java']=='Дизайн') ]['requirements']

txt1.head()
txt1 = txt1.dropna()
# Preprocessing 

def remove_string_special_characters(s): 

      

    # removes special characters with ' ' 

    stripped = re.sub('[^a-zA-z\s]', '', s) 

    # stripped = re.sub('[0-9]+', '', stripped)

    stripped = re.sub('_', '', stripped) 

      

    # Change any white space to one space 

    stripped = re.sub('\s+', ' ', stripped) 

      

    # Remove start and end white spaces 

    stripped = stripped.strip() 

    if stripped != '': 

            return stripped.lower() 
stop_words = set(stopwords.words('english')) 

your_list = ['skills', 'ability', 'job', 'description'] 

for i, line in enumerate(txt1): 

    txt1[i] = ' '.join([x for 

        x in nltk.word_tokenize(line) if 

        ( x not in stop_words ) and ( x not in your_list ) ]) 

      

# Getting trigrams  

vectorizer = CountVectorizer(ngram_range = (4,4)) 

X1 = vectorizer.fit_transform(txt1)  

features = (vectorizer.get_feature_names()) 

#print("\n\nFeatures : \n", features[:100]) 
# # Applying TFIDF 

# vectorizer = TfidfVectorizer(ngram_range = (3,3)) 

# X2 = vectorizer.fit_transform(txt1) 

# scores = (X2.toarray()) 

# # print("\n\nScores : \n", scores) 

  

# Getting top ranking features 

sums = X1.sum(axis = 0) 

data1 = [] 

for col, term in enumerate(features): 

    data1.append( (term, sums[0,col] )) 

ranking = pd.DataFrame(data1, columns = ['term','rank']) 

words = (ranking.sort_values('rank', ascending = False)) 

words.head(50)
words.to_csv('dou-BoW-4-grams-design.csv')