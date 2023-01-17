# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir='/kaggle/input/disaster-tweets/'
df = pd.read_csv(data_dir+'tweets.csv')
df.head(2)
print (f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns" )
print (f"Columns in the data set: {df.columns.values}")
print (f"There are   are {len(df.keyword.unique())} unique keywords.")
print (f"The unique keywords are \n: {df.keyword.unique()}")
import pylab as plt
df.keyword.value_counts().plot(kind='bar', color='steelblue', figsize=(20, 4))
plt.title('Value Counts of Different Keywords',fontsize=18)
plt.xticks(fontsize=16, rotation=90, color='steelblue');
import pylab as plt
plt.figure(figsize=(20,12))
plt.subplot(2,1,1)
df.keyword.value_counts()[:30].plot(kind='bar', color='steelblue')#, figsize=(20, 10))
plt.title('Value Counts for Top 30 Keywords',fontsize=18)
plt.xticks(fontsize=16, rotation=90, color='steelblue');
plt.subplots_adjust(hspace=0.8)

plt.subplot(2,1,2)
df.keyword.value_counts()[-30:].plot(kind='bar', color='steelblue')#, figsize=(20, 10))
plt.title('Value Counts for Bottom 30 Keywords',fontsize=18)
plt.xticks(fontsize=16, rotation=90, color='steelblue');
plt.subplots_adjust(wspace=0.5)
print ( df['text'][0], '\n\n', df['text'][123])
df[df['target']==1]['text'].values[123]
df[df['target']==0]['text'].values[123]
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)# delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in STOPWORDS]) # delete stopwors from text
    return text

df['text_processed']=df['text'].apply(lambda x:preprocess_text(x))
df.sample(2)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

def split_vectorize_text(df, vec_method='tfidf'):
    
    # Fun Fact: 8848 is the height of Mt Everest (Nepal) in meters.
    df_train_val, df_test  = train_test_split(df, test_size=0.2, random_state = 8848) 
    df_train    , df_valid = train_test_split(df_train_val, test_size=0.25, random_state = 8848)

    if vec_method=='tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.95)
    elif vec_method=='count':
        vectorizer = CountVectorizer(max_df=0.95)
        
    x_field = 'text_processed'
    y_field = 'target'    
    
    X_train = vectorizer.fit_transform(df_train[x_field])
    X_valid = vectorizer.transform(df_valid[x_field])
    X_test = vectorizer.transform(df_test[x_field])
    
    y_train = df_train[y_field]
    y_valid = df_valid[y_field]
    y_test  = df_test[y_field]
    
    return (X_train, y_train, X_valid, y_valid, X_test, y_test, vectorizer)

(X_train, y_train, X_valid, y_valid, X_test, y_test, vectorizer) = split_vectorize_text(df)

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import accuracy_score, f1_score 

(X_train, y_train, X_valid, y_valid, X_test, y_test, vectorizer) = split_vectorize_text(df)
    
def train_model(X_train, y_train, method='logistic_regression'):
    if method =='logistic_regression':
        log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=8848,max_iter=100)
        model   = log_reg.fit(X_train, y_train)

    elif method =='ridge_classifier':
        clf = RidgeClassifier(solver='auto',random_state=8848, max_iter=100)
        model   = clf.fit(X_train, y_train)

    elif method =='random_forest':        
        clf = RandomForestClassifier(max_depth=5, random_state=8848)        
        model   = clf.fit(X_train, y_train)

    return model

def calc_accuracy(model, X, y):
    preds   = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    return (100*np.round(acc, 2), 100*np.round(f1, 2) )
#methods = [LogisticRegression, RidgeClassifier]
results = []
methods = ['logistic_regression', 'ridge_classifier', 'random_forest']
for method in methods:
    model = train_model(X_train, y_train, method=method) 
    acc_train, f1_train= calc_accuracy(model, X_train, y_train)
    acc_valid, f1_valid= calc_accuracy(model, X_valid, y_valid)
    print (f" Method: {method}, \n Training data: accuracy: {acc_train}% and f1_score: {f1_train}% \n \
Validation data: accuracy:  {acc_valid}%  and f1_score: {f1_valid}% \n ")
    results.append( [method, acc_train, acc_valid, f1_train, f1_valid ] )
df_results = pd.DataFrame(data=results, columns=['method',
                                                 'Training Accuracy', 'Validation Accuracy', 
                                                 'F1 score (Train)', 'F1 score (Validation)'])
df_results.set_index('method')
results = []

for vm in ['tfidf', 'count']:
    (X_train, y_train, X_valid, y_valid, X_test, y_test, vectorizer) = split_vectorize_text(df, vec_method=vm)
    for method in ['logistic_regression', 'ridge_classifier', 'random_forest']:
        model = train_model(X_train, y_train, method=method) 
        acc_train, f1_train= calc_accuracy(model, X_train, y_train)
        acc_valid, f1_valid= calc_accuracy(model, X_valid, y_valid)
        print (f" Vectorizer: {vm}, Method: {method}, \n Training data: accuracy: {acc_train}% and f1_score: {f1_train}% \n \
    Validation data: accuracy:  {acc_valid}%  and f1_score: {f1_valid}% \n ")
        results.append( [method, vm, acc_train, acc_valid, f1_train, f1_valid ] )
    
    
df_results = pd.DataFrame(data=results, columns=['method', 'vectorizer',
                                                 'Training Accuracy', 'Validation Accuracy', 
                                                 'F1 score (Train)', 'F1 score (Validation)'])
df_results.set_index('method')
