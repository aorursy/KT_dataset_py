# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import  matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
df_train=pd.read_csv("../input/train.tsv",delimiter='\t')
df_test=pd.read_csv("../input/test.tsv",delimiter='\t')
submission = pd.read_csv("../input/sampleSubmission.csv")
# Any results you write to the current directory are saved as output.
df_train.info()
df_train.head()
df_train.shape
df_test.shape
df_train['Sentiment'].value_counts()
fig=plt.figure(figsize=(6,8))
df_train.groupby('Sentiment').Phrase.count().plot.bar(ylim=0)
plt.show()
df_train['Phrase'] = df_train['Phrase'].apply(lambda x: x.lower())
df_test['Phrase'] = df_test['Phrase'].apply(lambda x: x.lower())
df_train.head()
tfidf_words = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = 'english',ngram_range = (1,3), analyzer = 'word', encoding = 'utf-8')
tfidf_char = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = 'english',ngram_range = (2,6), analyzer = 'char', encoding = 'utf-8')
X_train_words=tfidf_words.fit_transform(df_train['Phrase'])
X_train_char=tfidf_char.fit_transform(df_train['Phrase'])
X_test_words=tfidf_words.transform(df_test['Phrase'])
X_test_char=tfidf_char.transform(df_test['Phrase'])
X_train = sparse.hstack([X_train_words, X_train_char])
X_test = sparse.hstack([X_test_words, X_test_char])
X_train.shape
X_test.shape
y_train=df_train['Sentiment']
model=LogisticRegression(multi_class='multinomial',solver='newton-cg')
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
#clf=GridSearchCV(model,param_grid,cv=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
submission['Sentiment'] = y_pred
submission.to_csv("submission_LogisticRegression.csv", index = False)


