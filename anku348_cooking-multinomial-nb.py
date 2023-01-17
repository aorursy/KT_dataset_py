# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.chdir("../input")
# Any results you write to the current directory are saved as output.
import json
file = 'train.json'
with open(file) as train_file:
    dict_train = json.load(train_file)
train = pd.DataFrame.from_dict(dict_train)
#train.head()
del train['id']
train['ingredients'] = train['ingredients'].apply(lambda x: ', '.join(x))
train.head()
train.cuisine.unique()
train.groupby('cuisine').count()
indian = train[train.cuisine=="indian"]["ingredients"].str.split(',',expand=True).unstack().value_counts()
indian = pd.DataFrame(indian)
indian = indian.reset_index()
indian.columns = ['Words','Frequency']
indian.head()
cuisine_names = train['cuisine'].unique()
print (cuisine_names)
cuisine_to_id = {}
assign_id = 0
for name in cuisine_names:
    cuisine_to_id[name] = assign_id
    assign_id += 1  ## Get a new id for new item
    
##  Print the dictionary created
for key, values in cuisine_to_id.items():
    print (key, values)
id_to_cuisine_name = {v: k for k, v in cuisine_to_id.items()}
for key, values in id_to_cuisine_name.items():
    print (key, values)
def get_cuisine_id(cuisine_name):
    return cuisine_to_id[cuisine_name]

train['cuisine_id'] = train['cuisine'].map(get_cuisine_id)
## Split the data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train['ingredients'],train['cuisine_id'], 
                                                    test_size=0.15, random_state=42)
print ("Training Sample Size:", len(X_train), ' ', "Test Sample Size:" ,len(X_test))
## Get the word vocabulary out of the data
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

## Count of 'mistak' in corpus (mistake -> mistak after stemming)
print ('water appears:', count_vect.vocabulary_.get(u'water') , 'in the corpus')
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
print ('Dimension of TF-IDF vector :' , X_train_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)
## Prediction part
# vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1, 2))
X_test_tfidf = vectorizer.transform(X_test)
print ('Dimension of TF-IDF vector :' , X_test_tfidf.shape)

predicted = clf.predict(X_test_tfidf)
## predictions for first 10 test samples

counter  = 0
for doc, category in zip(X_test, predicted):
    print('%r => %s' % (doc, id_to_cuisine_name[category]))
    if(counter == 10):
        break
    counter += 1 
import numpy as np
np.mean(predicted == y_test) ## 80% sounds good only 