# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import plotly.offline as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True) 

from plotly import tools

import plotly.figure_factory as ff



import nltk

import re

import string





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/spam.csv',delimiter=',',encoding='latin-1')
data.head()
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
py.iplot(ff.create_table(data.head()),filename='show_data')
data.columns=['label','text']

py.iplot(ff.create_table(data.head()),filename='show_data')
data.text[0]
dir(string)
string.punctuation
def remove_punctuation(text):

    new_text=''.join([char for char in text if char not in string.punctuation])

    return new_text
data['new_text']=data['text'].apply(lambda row : remove_punctuation(row))
data.head()
print(data.text[0])

data.new_text[0]
def tokenize(text):

    tokens=re.split('\W+',text)

    return tokens 
data['tokenized_text']=data['new_text'].apply(lambda row : tokenize(row.lower()))

data.head()
stopwords=nltk.corpus.stopwords.words('english')

stopwords[:5]
def remove_stopwords(text):

    clean_text=[word for word in text if word not in stopwords]

    return clean_text 
data['clean_text']=data['tokenized_text'].apply(lambda row : remove_stopwords(row))

data.head()
ps = nltk.PorterStemmer()
dir(ps)
def stemming(tokenized_text):

    stemmed_text=[ps.stem(word) for word in tokenized_text]

    return stemmed_text
data['stemmed_text']=data.clean_text.apply(lambda row : stemming(row))

data[['text','stemmed_text']].head()
def get_final_text(stemmed_text):

    final_text=" ".join([word for word in stemmed_text])

    return final_text
data['final_text']=data.stemmed_text.apply(lambda row : get_final_text(row))

data.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_model=TfidfVectorizer()

tfidf_vec=tfidf_model.fit_transform(data.final_text)

tfidf_data=pd.DataFrame(tfidf_vec.toarray())

tfidf_data.head()
def count_punct(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")), 3)*100
data['punct%'] = data['text'].apply(lambda x: count_punct(x))
bins = np.linspace(0, 100, 40)

plt.hist(data['punct%'], bins)

plt.title("Punctuation Distribution")

plt.show()
data['text_len'] = data['text'].apply(lambda x: len(x) - x.count(" "))
bins = np.linspace(0, 250, 40)

plt.hist(data['text_len'],bins)

plt.title("text Length Distribution")

plt.show()
final_data=pd.concat([data['punct%'],data['text_len'],tfidf_data],axis=1)

final_data.head()
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(final_data,data['label'],test_size=.2)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=50, max_depth=None, n_jobs=-1)

rf_model = rf.fit(X_train, y_train)
rf_prediction=rf_model.predict(X_test)
precision,recall,fscore,support = score(y_test,rf_prediction,pos_label='spam',average='binary')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),

                                                        round(recall, 3),

                                                        round((rf_prediction==y_test).sum() / len(rf_prediction),3)))
sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]
from sklearn.model_selection import GridSearchCV
rfg = RandomForestClassifier()

param = {'n_estimators': [10, 150, 300],

        'max_depth': [30, 60, 90, None]}



gs = GridSearchCV(rfg, param, cv=5, n_jobs=-1)

gs_fit = gs.fit(final_data, data['label'])
print(gs_fit.best_params_)

print(gs_fit.best_score_)