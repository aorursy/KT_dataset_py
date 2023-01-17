# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import sent_tokenize, word_tokenize 
disaster = train.loc[train['target'] == 1]['text'] 

not_disaster = train.loc[train['target'] == 0]['text']
vectorizer = CountVectorizer(min_df=0, lowercase=True)

vectorizer.fit(disaster)

disaster_voc = vectorizer.vocabulary_

vectorizer.fit(not_disaster)

not_disaster_voc = vectorizer.vocabulary_
def mean_dic(dic):

    count = 0

    _sum = 0

    for key in dic:

        count += 1

        _sum += dic[key]

        return _sum / count
med_1 = np.quantile(list(disaster_voc.values()), .50)

med_2 = np.quantile(list(not_disaster_voc.values()), .50)

q_75_1 = np.quantile(list(disaster_voc.values()), .75)

q_75_2  = np.quantile(list(not_disaster_voc.values()), .75)
total_scores = []

avg_total_scores = []

pos_counts = []

avg_pos_counts = []

neg_counts = []

avg_neg_counts = []

ratios = []

avg_ratios = []



med_total_scores = []

med_pos_counts = []

med_ratios = []

med_neg_counts = []



q75_total_scores = []

q75_pos_counts = []

q75_ratios = []

q75_neg_counts = []





for i in range(len(train['text'].values)):

    total_score =  0

    avg_total_score = 0

    pos_count = 0

    avg_pos_count = 0

    neg_count = 0

    avg_neg_count = 0

    pos_score = 0

    avg_pos_score = 0

    neg_score = 0

    avg_neg_score = 0

    

    med_total_score = 0

    med_pos_count = 0

    med_pos_score = 0

    med_neg_count = 0

    med_neg_score = 0

    

    q75_total_score = 0

    q75_pos_count = 0

    q75_pos_score = 0

    q75_neg_count = 0

    q75_neg_score = 0

    for common_key in set(disaster_voc) & set([x.lower() for x in word_tokenize(train['text'].values[i])]):

        total_score += disaster_voc[common_key]

        pos_count += 1

        pos_score += disaster_voc[common_key]

        if disaster_voc[common_key] > mean_dic(disaster_voc):

            avg_total_score += disaster_voc[common_key]

            avg_pos_count += 1

            avg_pos_score += disaster_voc[common_key]

        if disaster_voc[common_key] > med_1:

            med_total_score += disaster_voc[common_key]

            med_pos_count += 1

            med_pos_score += disaster_voc[common_key]

        if disaster_voc[common_key] > q_75_1:

            q75_total_score += disaster_voc[common_key]

            q75_pos_count += 1

            q75_pos_score += disaster_voc[common_key]

    for common_key in set(not_disaster_voc) & set([x.lower() for x in word_tokenize(train['text'].values[i])]):

        total_score -= not_disaster_voc[common_key]

        neg_count += 1

        neg_score += not_disaster_voc[common_key]

        if not_disaster_voc[common_key] > mean_dic(not_disaster_voc):

            avg_total_score -= not_disaster_voc[common_key]

            avg_neg_count += 1

            avg_neg_score += not_disaster_voc[common_key]

        if not_disaster_voc[common_key] > med_2:

            med_total_score -= not_disaster_voc[common_key]

            med_neg_count += 1

            med_neg_score += not_disaster_voc[common_key]

        if not_disaster_voc[common_key] > q_75_2:

            q75_total_score -= not_disaster_voc[common_key]

            q75_neg_count += 1

            q75_neg_score += not_disaster_voc[common_key]

            

            

    

    total_scores.append(total_score)

    pos_counts.append(pos_count)

    neg_counts.append(neg_count)

    ratios.append(pos_score / (neg_score + 0.001))

    

    avg_total_scores.append(total_score)

    avg_pos_counts.append(pos_count)

    avg_neg_counts.append(neg_count)

    avg_ratios.append(avg_pos_score / (avg_neg_score + 0.001))

    



    med_total_scores.append(med_total_score)

    med_pos_counts.append(med_pos_count)

    med_neg_counts.append(med_neg_count)

    med_ratios.append(med_pos_score / (med_neg_score + 0.001))

    

    q75_total_scores.append(q75_total_score)

    q75_pos_counts.append(q75_pos_count)

    q75_neg_counts.append(q75_neg_count)

    q75_ratios.append(q75_pos_score / (q75_neg_score + 0.001))

    
train['total_scores'] =  total_scores

train['pos_counts'] =  pos_counts

train['neg_counts'] =  neg_counts

train['ratios'] =  ratios



train['avg_total_scores'] =  avg_total_scores

train['avg_pos_counts'] =  avg_pos_counts

train['avg_neg_counts'] =  avg_neg_counts

train['avg_ratios'] =  avg_ratios



train['med_total_scores'] =  med_total_scores

train['med_pos_counts'] =  med_pos_counts

train['med_neg_counts'] =  med_neg_counts

train['med_ratios'] =  med_ratios



train['q75_total_scores'] =  q75_total_scores

train['q75_pos_counts'] =  q75_pos_counts

train['q75_neg_counts'] =  q75_neg_counts

train['q75_ratios'] =  q75_ratios
total_scores = []

avg_total_scores = []

pos_counts = []

avg_pos_counts = []

neg_counts = []

avg_neg_counts = []

ratios = []

avg_ratios = []



med_total_scores = []

med_pos_counts = []

med_ratios = []

med_neg_counts = []



q75_total_scores = []

q75_pos_counts = []

q75_ratios = []

q75_neg_counts = []





for i in range(len(test['text'].values)):

    total_score =  0

    avg_total_score = 0

    pos_count = 0

    avg_pos_count = 0

    neg_count = 0

    avg_neg_count = 0

    pos_score = 0

    avg_pos_score = 0

    neg_score = 0

    avg_neg_score = 0

    

    med_total_score = 0

    med_pos_count = 0

    med_pos_score = 0

    med_neg_count = 0

    med_neg_score = 0

    

    q75_total_score = 0

    q75_pos_count = 0

    q75_pos_score = 0

    q75_neg_count = 0

    q75_neg_score = 0

    for common_key in set(disaster_voc) & set([x.lower() for x in word_tokenize(test['text'].values[i])]):

        total_score += disaster_voc[common_key]

        pos_count += 1

        pos_score += disaster_voc[common_key]

        if disaster_voc[common_key] > mean_dic(disaster_voc):

            avg_total_score += disaster_voc[common_key]

            avg_pos_count += 1

            avg_pos_score += disaster_voc[common_key]

        if disaster_voc[common_key] > med_1:

            med_total_score += disaster_voc[common_key]

            med_pos_count += 1

            med_pos_score += disaster_voc[common_key]

        if disaster_voc[common_key] > q_75_1:

            q75_total_score += disaster_voc[common_key]

            q75_pos_count += 1

            q75_pos_score += disaster_voc[common_key]

    for common_key in set(not_disaster_voc) & set([x.lower() for x in word_tokenize(test['text'].values[i])]):

        total_score -= not_disaster_voc[common_key]

        neg_count += 1

        neg_score += not_disaster_voc[common_key]

        if not_disaster_voc[common_key] > mean_dic(not_disaster_voc):

            avg_total_score -= not_disaster_voc[common_key]

            avg_neg_count += 1

            avg_neg_score += not_disaster_voc[common_key]

        if not_disaster_voc[common_key] > med_2:

            med_total_score -= not_disaster_voc[common_key]

            med_neg_count += 1

            med_neg_score += not_disaster_voc[common_key]

        if not_disaster_voc[common_key] > q_75_2:

            q75_total_score -= not_disaster_voc[common_key]

            q75_neg_count += 1

            q75_neg_score += not_disaster_voc[common_key]

            

            

    

    total_scores.append(total_score)

    pos_counts.append(pos_count)

    neg_counts.append(neg_count)

    ratios.append(pos_score / (neg_score + 0.001))

    

    avg_total_scores.append(total_score)

    avg_pos_counts.append(pos_count)

    avg_neg_counts.append(neg_count)

    avg_ratios.append(avg_pos_score / (avg_neg_score + 0.001))

    



    med_total_scores.append(med_total_score)

    med_pos_counts.append(med_pos_count)

    med_neg_counts.append(med_neg_count)

    med_ratios.append(med_pos_score / (med_neg_score + 0.001))

    

    q75_total_scores.append(q75_total_score)

    q75_pos_counts.append(q75_pos_count)

    q75_neg_counts.append(q75_neg_count)

    q75_ratios.append(q75_pos_score / (q75_neg_score + 0.001))

    
test['total_scores'] =  total_scores

test['pos_counts'] =  pos_counts

test['neg_counts'] =  neg_counts

test['ratios'] =  ratios



test['avg_total_scores'] =  avg_total_scores

test['avg_pos_counts'] =  avg_pos_counts

test['avg_neg_counts'] =  avg_neg_counts

test['avg_ratios'] =  avg_ratios



test['med_total_scores'] =  med_total_scores

test['med_pos_counts'] =  med_pos_counts

test['med_neg_counts'] =  med_neg_counts

test['med_ratios'] =  med_ratios



test['q75_total_scores'] =  q75_total_scores

test['q75_pos_counts'] =  q75_pos_counts

test['q75_neg_counts'] =  q75_neg_counts

test['q75_ratios'] =  q75_ratios
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import balanced_accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
train
X = train.drop(['keyword','location','target','text'],axis = 1)

y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
classifier = RandomForestClassifier(n_estimators = 100,max_features = 'auto',max_depth = 5,criterion = 'gini',

                                    min_samples_split= 15,min_samples_leaf = 5,max_leaf_nodes = 10)

classifier.fit(X, y)

predictions = classifier.predict(X)
balanced_accuracy_score(predictions,y)
classifier = RandomForestClassifier(n_estimators = 100,max_features = 'auto',max_depth = 5,criterion = 'gini',

                                    min_samples_split= 15,min_samples_leaf = 5,max_leaf_nodes = 10)

classifier.fit(X, y)

predictions = classifier.predict(test.drop(['keyword','location','text'],axis = 1))
submission['target'] = predictions
submission.to_csv('submission.csv',index = False)