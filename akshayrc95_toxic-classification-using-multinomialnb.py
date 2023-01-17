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
df=pd.read_csv('../input/train.csv')
removelist = "\'"
df = df.replace('\n',' ', regex=True)
df = df.replace(r'[^\w'+removelist+']',' ', regex=True)
df
dft=pd.read_csv('../input/test.csv')
removelist = "\'"
dft = dft.replace('\n',' ', regex=True)
dft = dft.replace(r'[^\w'+removelist+']',' ', regex=True)
dft
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
text_clf1 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])
text_clf1 = text_clf1.fit(df['comment_text'], df['toxic'])
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3,1e-3),
}
gs_clf1 = GridSearchCV(text_clf1, parameters, n_jobs=-1)
gs_clf1 = gs_clf1.fit(df['comment_text'], df['toxic'])
#print(gs_clf1.best_score_)
pred1=gs_clf1.predict_proba(dft['comment_text'])
ans1=[round(pred1[i][1],15) for i in range(len(pred1))]

text_clf2 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])
text_clf2 = text_clf2.fit(df['comment_text'], df['severe_toxic'])
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}
gs_clf2 = GridSearchCV(text_clf2, parameters, n_jobs=-1)
gs_clf2 = gs_clf2.fit(df['comment_text'], df['severe_toxic'])
print(gs_clf2.best_score_)
pred2=gs_clf2.predict_proba(dft['comment_text'])
ans2=[pred2[i][1] for i in range(len(pred2))]

text_clf3 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])
text_clf3 = text_clf3.fit(df['comment_text'], df['obscene'])
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}
gs_clf3 = GridSearchCV(text_clf3, parameters, n_jobs=-1)
gs_clf3 = gs_clf3.fit(df['comment_text'], df['obscene'])
print(gs_clf3.best_score_)
pred3=gs_clf3.predict_proba(dft['comment_text'])
ans3=[pred3[i][1] for i in range(len(pred3))]

text_clf4 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])
text_clf4 = text_clf4.fit(df['comment_text'], df['threat'])
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}
gs_clf4 = GridSearchCV(text_clf4, parameters, n_jobs=-1)
gs_clf4 = gs_clf4.fit(df['comment_text'], df['threat'])
print(gs_clf4.best_score_)
pred4=gs_clf4.predict_proba(dft['comment_text'])
ans4=[pred4[i][1] for i in range(len(pred4))]

text_clf5 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])
text_clf5 = text_clf5.fit(df['comment_text'], df['insult'])
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}
gs_clf5 = GridSearchCV(text_clf5, parameters, n_jobs=-1)
gs_clf5 = gs_clf5.fit(df['comment_text'], df['insult'])
print(gs_clf5.best_score_)
pred5=gs_clf5.predict_proba(dft['comment_text'])
ans5=[pred5[i][1] for i in range(len(pred5))]

text_clf6 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])
text_clf6 = text_clf6.fit(df['comment_text'], df['identity_hate'])
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
}
gs_clf6 = GridSearchCV(text_clf6, parameters, n_jobs=-1)
gs_clf6 = gs_clf6.fit(df['comment_text'], df['identity_hate'])
print(gs_clf6.best_score_)
pred6=gs_clf6.predict_proba(dft['comment_text'])
ans6=[pred6[i][1] for i in range(len(pred6))]

dic={}
dic['id']=dft['id']
dic['toxic']=ans1
dic['severe_toxic']=ans2
dic['obscene']=ans3
dic['threat']=ans4
dic['insult']=ans5
dic['identity_hate']=ans6
ans=pd.DataFrame(dic)

ans = ans[['id', 'toxic','severe_toxic','obscene','threat','insult','identity_hate']]

ans.iloc[:,].to_csv('submission.csv',sep=',',index = False,float_format='%.15f')