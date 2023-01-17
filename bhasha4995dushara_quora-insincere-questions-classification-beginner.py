import os
print(os.listdir("../input"))
!ls ../input/embeddings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


import nltk
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from string import punctuation as str_pun
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
train = pd.read_csv('../input/train.csv')
print(train.shape)
train.head()
test = pd.read_csv('../input/test.csv')
print(test.shape)
test.head()
insincere  = train[train['target']==1]
print("length of the INSINCERE : ",len(insincere))
sincere = train[train['target']==0]
print("length of the SINCERE :",len(sincere))
sns.countplot(data=train,hue=train['target'],x=train['target'])
train["num_words"] = train["question_text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["question_text"].apply(lambda x: len(str(x).split()))
train["num_unique_words"] = train["question_text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["question_text"].apply(lambda x: len(set(str(x).split())))
train["num_chars"] = train["question_text"].apply(lambda x: len(str(x)))
test["num_chars"] = test["question_text"].apply(lambda x: len(str(x)))
w = stopwords.words('english')
train['num_stopwords'] = train['question_text'].apply(lambda x : len([nw for nw in str(x).split() if nw.lower() in w]))
test['num_stopwords'] = test['question_text'].apply(lambda x : len([nw for nw in str(x).split() if nw.lower() in w]))
train['num_punctuation'] = train['question_text'].apply(lambda x : len([np for np in str(x) if np in str_pun]))
test['num_punctuation'] = test['question_text'].apply(lambda x : len([np for np in str(x) if np in str_pun]))
train['num_uppercase'] = train['question_text'].apply(lambda x : len([nu for nu in str(x).split() if nu.isupper()]))
test['num_uppercase'] = test['question_text'].apply(lambda x : len([nu for nu in str(x).split() if nu.isupper()]))
train['num_lowercase'] = train['question_text'].apply(lambda x : len([nl for nl in str(x).split() if nl.islower()]))
test['num_lowercase'] = test['question_text'].apply(lambda x : len([nl for nl in str(x).split() if nl.islower()]))
train['num_title'] = train['question_text'].apply(lambda x : len([nl for nl in str(x).split() if nl.istitle()]))
test['num_title'] = test['question_text'].apply(lambda x : len([nl for nl in str(x).split() if nl.istitle()]))
train[train['target']==1].describe()
train[train['target']==0].describe()
sns.violinplot(x=train['target'],y=train['num_chars'],data=train)
sns.violinplot(x=train['target'],y=train['num_words'],data=train,split=True)
sns.violinplot(x='target',y='num_unique_words',data=train,split=True)
plt.figure(figsize=(20,15))
sns.stripplot(x='num_words',y='num_unique_words',data=train, hue='target',jitter=False)#, split=True)
sns.stripplot(x='target',y='num_stopwords',data=train, jitter=False)
def text_process(question):
    nopunc = [char for char in question if char not in str_pun]
    nopunc = "".join(nopunc)
    meaning = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return( " ".join( meaning )) 
print("Processing ...")
train['question_text'].apply(text_process)
print("processing ...")
test['question_text'].apply(text_process)
train.head()
pipeline = Pipeline([('cv',CountVectorizer(analyzer='word',ngram_range=(1,4),max_df=0.9)),
                     ('clf',LogisticRegression(solver='saga',class_weight='balanced',C=0.45,max_iter=250, verbose=1))])
pipeline.get_params().keys()
X_train = train['question_text'].values
y_train = train['target']
X_test = test['question_text'].values
pipeline.fit(X_train,y_train)
prediction = pipeline.predict(X_test)
insincere = prediction[prediction == 1]
print("Length of INSINCERE after Prediction : ",len(insincere))
sincere = prediction[prediction == 0]
print("Length of SINCERE after Prediction : ",len(sincere))
submit = pd.DataFrame({'qid':test['qid'],'prediction':prediction})
submit.head()
submit.to_csv('submission.csv',index=False)
