# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import pandas as pd

import numpy as np

import json

import re

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix





from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = []



for line in open(r'/kaggle/input/news-classification/News Classification DataSet.json', 'r'):

    data.append(json.loads(line))



content, label = [], []

for each in data:

    content.append(each['content'])

    label.append(each['annotation']['label'][0])

    

df = pd.DataFrame([content, label]).T

df.columns= ['content', 'label']

df.head()
stopwords_english = set(stopwords.words('english'))

df['content'] = df.content.apply(lambda x:str(x))

def remove_stopwords(s):

    return ' '.join(word for word in s.split() if word not in stopwords_english)

df['content'] = df.content.apply(lambda x:remove_stopwords(x))

df.loc[:,'content']  = df.content.apply(lambda x:" ".join(re.findall('[\w]+',x)))
df.groupby('label').size()
texts = df['content'].astype('str')



tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), 

                                   min_df = 2, 

                                   max_df = .95)



X = tfidf_vectorizer.fit_transform(texts) #features

y = df['label'].values #target



print (X.shape)

print(y.shape)
lsa = TruncatedSVD(n_components=100, 

                   n_iter=10, 

                   random_state=3)



X = lsa.fit_transform(X)

X.shape
#Preliminary model evaluation using default parameters



#Creating a dict of the models

model_dict = {'Stochastic Gradient Descent' : SGDClassifier(random_state=3, loss='log'),

              'Random Forest': RandomForestClassifier(n_estimators=100,random_state=3),

              'Decsision Tree': DecisionTreeClassifier(random_state=3),

              'AdaBoost': AdaBoostClassifier(random_state=3),

              'Gaussian Naive Bayes': GaussianNB()}



#Train test split with stratified sampling for evaluation

X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size = .3, 

                                                    shuffle = True, 

                                                    stratify = y, 

                                                    random_state = 3)



#Function to get the scores for each model in a df

def model_score_df(model_dict):   

    model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], []

    for k,v in model_dict.items():   

        model_name.append(k)

        v.fit(X_train, y_train)

        y_pred = v.predict(X_test)

        ac_score_list.append(accuracy_score(y_test, y_pred))

        p_score_list.append(precision_score(y_test, y_pred, average='macro'))

        r_score_list.append(recall_score(y_test, y_pred, average='macro'))

        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))

        model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T

        model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']

        model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)

    return model_comparison_df



model_score_df(model_dict)