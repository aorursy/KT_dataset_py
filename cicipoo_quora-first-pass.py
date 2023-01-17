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
train = pd.read_csv('../input/train.csv')
df = train.copy()
train.head()
sample = pd.read_csv('../input/sample_submission.csv')
sample.head()
train.groupby('target').count()
train.groupby('target').count() / train.shape[0] * 100
list(train.loc[train.target == 1].question_text.head())
list(train.loc[train.target == 1].question_text.sample(10,random_state=504))
list(train.loc[train.target == 0].question_text.sample(15,random_state=405))
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from stop_words import get_stop_words
import re

stops = get_stop_words('english')
def clean_list(x):
    x = [i.strip() for i in x if i.strip() not in stops and i.strip() != '']
    return x
    
train.question_text = train.question_text.apply(lambda x: x.lower().strip())
train.question_text = train.question_text.apply(lambda x: re.sub(r'[?,\.!\"\']',' ', x))
train.question_text = train.question_text.apply(lambda x: x.split(' '))
train.question_text = train.question_text.apply(clean_list)
train.head()
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
train.question_text = train.question_text.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
train['preprocessed_text'] = train.question_text.apply(lambda x: ' '.join(x))
train.question_text = df.question_text
train.head()
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
train_pos = train.loc[train.target == 1].sample(5000,random_state=5094)
train_neg = train.loc[train.target == 0].sample(5000,random_state=5094)
test_pos = train.loc[(train.target == 1) & ~train.qid.isin(train_pos.qid)].sample(5000,random_state=5094)
test_neg = train.loc[(train.target == 0) & ~train.qid.isin(train_neg.qid)].sample(5000,random_state=5094)
train_subset = pd.concat([train_pos,train_neg,test_pos,test_neg])
train_subset['ner_data'] = train_subset.question_text.apply(lambda x: nlp(x).ents)
def get_ne_counts(tuples):
    nes = {}
    for tup in tuples:
        info = (tup.text, tup.label_)
        if info[1] in nes:
            nes[info[1]] += 1
        else:
            nes[info[1]] = 1
    return nes

train_subset['ne_types'] = train_subset.ner_data.apply(get_ne_counts)
ne_df = pd.DataFrame(list(train_subset['ne_types']),index=train_subset.index)
ne_df = ne_df.fillna(0)
train_subset = pd.concat([train_subset,ne_df],axis=1,join_axes=[train_subset.index])
train_subset.head()
cvec = CountVectorizer()
counts = cvec.fit_transform(train_subset.preprocessed_text)
counts = pd.DataFrame(counts.todense(),index=train_subset.index)
counts.columns = cvec.get_feature_names()
counts = counts[[c for c in counts.columns if counts[c].sum() >= 3]]
counts.head()
counts.to_csv('counts.csv',index=True,header=True)
feature_df = pd.concat([train_subset.iloc[:,6:],counts],axis=1)
feature_df['target'] = list(train_subset['target'])
feature_df.head()
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=40938)

X = feature_df[[c for c in feature_df.columns if c != 'target']]
y = feature_df.target
X_train = X.iloc[:10000,:]
X_test = X.iloc[10000:,:]
y_train = y.iloc[:10000]
y_test = y.iloc[10000:]

rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
eval_df = pd.DataFrame({'actual': list(y_test),
                       'predicted': predictions})
eval_df['incorrect'] = eval_df['actual'] - eval_df['predicted']
eval_df.incorrect = eval_df.incorrect.apply(abs)
1 - eval_df.incorrect.sum() / eval_df.shape[0]
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rfc, n_iter=1,param_distributions = random_grid, 
                               cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = sum(abs(predictions - test_labels)) / len(test_labels)
    accuracy = 100 - errors
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
predictions = best_random.predict(X_test)
eval_df = pd.DataFrame({'actual': list(y_test),
                       'predicted': predictions})
eval_df['incorrect'] = eval_df['actual'] - eval_df['predicted']
eval_df.incorrect = eval_df.incorrect.apply(abs)
1 - eval_df.incorrect.sum() / eval_df.shape[0]
precision = eval_df.loc[(eval_df.actual == 1) & (eval_df.predicted == 1)].shape[0] / \
eval_df.loc[(eval_df.predicted == 1)].shape[0]
precision
recall = eval_df.loc[(eval_df.actual == 1) & (eval_df.predicted == 1)].shape[0] / \
eval_df.loc[(eval_df.actual == 1)].shape[0]
recall
test_data = pd.read_csv('../input/test.csv')
test_data.head()
original = list(test_data['question_text'])
test_data.question_text = test_data.question_text.apply(lambda x: x.lower().strip())
test_data.question_text = test_data.question_text.apply(lambda x: re.sub(r'[?,\.!\"\']',' ', x))
test_data.question_text = test_data.question_text.apply(lambda x: x.split(' '))
test_data.question_text = test_data.question_text.apply(clean_list)
test_data.question_text = test_data.question_text.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
test_data['preprocessed_text'] = test_data.question_text.apply(lambda x: ' '.join(x))
test_data.question_text = original
test_data.head()
test_data['ner_data'] = test_data.question_text.apply(lambda x: nlp(x).ents)
test_data['ne_types'] = test_data.ner_data.apply(get_ne_counts)
ne_df = pd.DataFrame(list(test_data['ne_types']),index=test_data.index)
ne_df = ne_df.fillna(0)
test_data = pd.concat([test_data,ne_df],axis=1,join_axes=[test_data.index])
test_data.head()
counts = cvec.transform(test_data.preprocessed_text)
import gc

predictions = []
for i in range(5000,counts.shape[0]+4999,5000):
    gc.collect()
    subset = counts[i-5000:i].todense()
    subset = pd.DataFrame(subset,index=test_data.index[i-5000:i])
    subset.columns = cvec.get_feature_names()
    subset = subset[[c for c in subset.columns if c in X_train.columns]]
    feature_df = pd.concat([test_data.iloc[i-5000:i,6:],subset],axis=1)
    for col in X_train.columns:
        if col not in feature_df.columns:
            feature_df[col] = 0
    predictions += list(best_random.predict(feature_df))
test_data = pd.read_csv('../input/test.csv')
test_data['prediction'] = predictions
test_data.head()
test_data.groupby('prediction').count()
test_data = test_data[['qid','question_text']]
test_data.to_csv('submission.csv',index=None)
