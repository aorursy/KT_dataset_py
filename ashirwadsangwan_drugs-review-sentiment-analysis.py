# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings as fw
fw('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/drugsComTrain_raw.csv')
test = pd.read_csv('../input/drugsComTest_raw.csv')
train.head()
top_30_drugs = train.drugName.value_counts()[:30]
plt.figure(figsize = (15,7))
top_30_drugs.plot(kind = 'bar');
plt.title('Top 30 Drugs by Count',fontsize = 20);
top_30_problems = train.condition.value_counts()[:30]
plt.figure(figsize = (15,7))
top_30_problems.plot(kind = 'bar');
plt.title('Top 30 Problems',fontsize = 20);
import string
train['review_clean']=train['review'].str.replace('[{}]'.format(string.punctuation), '')
train.head()
train = train.fillna({'review':''})  # fill in N/A's in the review column
plt.figure(figsize = (15,7))
train.rating.value_counts().plot(kind = 'bar');
plt.xlabel('Ratings',fontsize = 15);
plt.title('Ratings by count',fontsize = 18);
train['sentiment'] = train['rating'].apply(lambda rating : +1 if rating > 5 else -1)
train.head()
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(train,test_size = 0.20)
print('Size of train_data is :', train_data.shape)
print('Size of test_data is :', test_data.shape)
import gc
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer()

train_matrix = vectorizer.transform(train_data['review_clean'].values.astype('U'))
test_matrix = vectorizer.transform(test_data['review_clean'].values.astype('U'))

gc.collect()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
rf = clf.fit(train_matrix,train_data['sentiment'])
y_pred = rf.predict(test_matrix)
from sklearn.metrics import f1_score
f1_score(y_pred,test_data.sentiment)
from sklearn import tree
import graphviz


clf = tree.DecisionTreeClassifier() # init the tree
clf = clf.fit(train_matrix, train_data.sentiment) # train the tree
# export the learned decision tree
dot_data = tree.export_graphviz(clf,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("sentiment") 
