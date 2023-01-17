# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import torch
import json
train=pd.read_json(r'../input/train.json')
test=pd.read_json(r'../input/test.json')
train.head()
train['ingredients_unrolled']=train['ingredients'].apply(lambda x :','.join(x))
train.head()
# finding unique words in the ingredients
unique_words=dict()
for row in train['ingredients_unrolled']:
    for w in row.split(','):
        if w not in unique_words:
            unique_words[w]=1
        else:
            unique_words[w]+=1
len(unique_words)
words_count=pd.Series(unique_words)
words_count.head()
from string import digits
#remove digits and percentage symbols from the text
s='abc123%'
digits+='%()oz'
print(digits)
remove_digits=str.maketrans('','',digits)
def clean_numbers_and_percentage(X):
    return X.translate(remove_digits)
train['ingredients_unrolled_cleaned']=train['ingredients_unrolled'].apply(clean_numbers_and_percentage)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(tokenizer=lambda x:x.split(','))
ing_vectors=tfidf.fit_transform(train['ingredients_unrolled'])
tfidf.get_feature_names()
tfidf=TfidfVectorizer(tokenizer=lambda x:x.split(','))
ing_vectors=tfidf.fit_transform(train['ingredients_unrolled_cleaned']).toarray()
tfidf.get_feature_names()[-100:]
tfidf.get_feature_names()[-1000:]
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
cusine_count=train['cuisine'].value_counts()
import matplotlib.pyplot as plt
plt.bar(cusine_count.index,cusine_count.values)
plt.title('Cusines count')
plt.xlabel('cusines')
plt.ylabel('count')
plt.xticks(rotation='vertical')
plt.show()

train['cuisine_encoded']=train['cuisine'].astype('category').cat.codes
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
ing_dim2=pca.fit_transform(ing_vectors)
plt.scatter(ing_dim2[:,0],ing_dim2[:,1],c=train['cuisine_encoded'])
plt.xlabel('pca 1')
plt.ylabel('pca 2')
plt.colorbar()
plt.show()
label_map=cusine_count.to_dict()
label_map
y_train=train['cuisine'].map(label_map)
from sklearn.model_selection import cross_val_score,cross_validate
scores=cross_validate(RandomForestClassifier(n_estimators=100),X=ing_vectors,y=y_train,cv=3)