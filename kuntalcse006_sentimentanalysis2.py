# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import string
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from pprint import pprint
from time import time
# import linear model classifier
from sklearn.linear_model import LogisticRegression
# import grid search class for perform grid search
from sklearn.grid_search import GridSearchCV
# import pipline for pipline class
from sklearn.pipeline import Pipeline
# take pipline as pipline instance
pipline = Pipeline([
                ('vect',CountVectorizer()),
                ('tfidf',TfidfTransformer()),
                ('lgr',LogisticRegression(solver='sag'))
              
])
# here we craete the pipline for entire text processing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
df_test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
df_train.head()
df_test.head()
col_test_name = ["id","comment_text"]
df_test_feature = df_test[col_test_name]
df_test_feature.shape
def replace_string(s):
    s = str(s)
    s = re.sub(r'[^\w\s\d+]','',s.lower())
    ss = re.sub('\d+','',s)
    text = re.sub('\n',' ',ss)
    return text
t0 = time()
df_train['comment_string'] = df_train.comment_text.apply(replace_string)
print('done in %0.3fs'%(time()-t0))
df_train.head()
t0 = time()
df_test_feature['comment_string'] = df_test_feature.comment_text.apply(replace_string)
print('done in %0.3fs'%(time()-t0))
df_test_feature.head()
xcol_name = ["id","comment_string"]
df_feature = df_train[xcol_name]
ycol_name = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
df_label = df_train[ycol_name]
#train_x, test_x, train_y, test_y = train_test_split(df_feature,df_label)
df_feature.shape,df_label.shape, df_test_feature.shape
# create parameter for grid search
parameter = {'tfidf__use_idf':(True,False)}
# perfrom grid search with pipline and parameter
model = GridSearchCV(pipline, parameter, n_jobs=-1, verbose=1)
print('Perform grid search now....')
print('Parameter :')
pprint(parameter)
t0 = time()
model.fit(df_feature.comment_string,df_label.toxic)
print('done in %0.3fs'%(time()-t0))
probability = model.predict_proba(df_test_feature.comment_string)
toxiclist = []
for i in range(len(probability)):
       toxiclist.append(probability[i][1])
# perfrom grid search with pipline and parameter
model = GridSearchCV(pipline, parameter, n_jobs=-1, verbose=1)
print('Perform grid search now....')
print('Parameter :')
pprint(parameter)
t0 = time()
model.fit(df_feature.comment_string,df_label.severe_toxic)
print('done in %0.3fs'%(time()-t0))
probability = model.predict_proba(df_test_feature.comment_string)
severetoxiclist = []
for i in range(len(probability)):
       severetoxiclist.append(probability[i][1])
model = GridSearchCV(pipline, parameter, n_jobs=-1, verbose=1)
print('Perform grid search now....')
print('Parameter :')
pprint(parameter)
t0 = time()
model.fit(df_feature.comment_string,df_label.obscene)
probability = model.predict_proba(df_test_feature.comment_string)
obscenelist = []
for i in range(len(probability)):
       obscenelist.append(probability[i][1])
print('done in %0.3fs'%(time()-t0))
model = GridSearchCV(pipline, parameter, n_jobs=-1, verbose=1)
print('Perform grid search now....')
print('Parameter :')
pprint(parameter)
t0 = time()
model.fit(df_feature.comment_string,df_label.threat)
probability = model.predict_proba(df_test_feature.comment_string)
threatlist = []
for i in range(len(probability)):
       threatlist.append(probability[i][1])
print('done in %0.3fs'%(time()-t0))
model = GridSearchCV(pipline, parameter, n_jobs=-1, verbose=1)
print('Perform grid search now....')
print('Parameter :')
pprint(parameter)
t0 = time()
model.fit(df_feature.comment_string,df_label.insult)
probability = model.predict_proba(df_test_feature.comment_string)
insultlist = []
for i in range(len(probability)):
       insultlist.append(probability[i][1])
print('done in %0.3fs'%(time()-t0))
model = GridSearchCV(pipline, parameter, n_jobs=-1, verbose=1)
print('Perform grid search now....')
print('Parameter :')
pprint(parameter)
t0 = time()
model.fit(df_feature.comment_string,df_label.identity_hate)
probability = model.predict_proba(df_test_feature.comment_string)
identity_hatelist = []
for i in range(len(probability)):
       identity_hatelist.append(probability[i][1])
print('done in %0.3fs'%(time()-t0))
#df_label.columns
mytoxicsubmision = pd.DataFrame({
    'id':list(df_test_feature.id),
    'toxic':toxiclist,
    'severe_toxic':severetoxiclist,
    'obscene':obscenelist,
    'threat':threatlist,
    'insult':insultlist,
    'identity_hate':identity_hatelist
})
mytoxicsubmision.columns = ["id","toxic","severe_toxic","obscene","threat","insult","identity_hate"]
mytoxicsubmision.shape
mytoxicsubmision.to_csv('mylasttoxicsubmission.csv', index=False) 

