# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set(color_codes=True)

sns.set_style("white")



from plotly.offline import plot

import plotly.graph_objs as go



import sklearn.ensemble

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, log_loss



import re

import nltk

from nltk.corpus import stopwords

import string

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer



stops = set(stopwords.words("english"))
train_set = pd.read_csv('../input/train.csv')



test_set = pd.read_csv('../input/test.csv')



print('There are {} records in train'.format(train_set.shape[0]))

print('There are {} records in train'.format(test_set.shape[0]))
target = 'is_duplicate'

ID = 'id'
train_set['question1'] = train_set['question1'].fillna('')

train_set['question2'] = train_set['question2'].fillna('')



test_set['question1'] = test_set['question1'].fillna('')

test_set['question2'] = test_set['question2'].fillna('')
def clean_text(text):

    text = re.sub('\s+', ' ', text)

    text = re.sub("\.\s", '.', text)

    text = re.sub(':\)', '.', text)

    text = re.sub("^\s\w+", '\w+', text)

    text = re.sub('\.', ' ', text)

    return text
def replace(sentence):

    sentence = sentence.replace('what\'s', 'what is').replace('don\'t', 'do not')

    sentence = sentence.replace('i\'m', 'i am').replace('can\'t', 'can not')

    sentence = sentence.replace('doesn\'t', 'does not').replace('it\'s', 'it is')

    sentence = sentence.replace('didn\'t', 'did not').replace('isn\'t', 'is not')

    sentence = sentence.replace('won\'t', 'will not').replace('aren\'t', 'are not')

    sentence = sentence.replace('shouldn\'t', 'should not').replace('haven\'t', 'have not')

    sentence = sentence.replace('hasn\'t', 'has not').replace('he\'s', 'he is')

    sentence = sentence.replace('wouldn\'t', 'would not').replace('he\'s', 'he is')

    sentence = sentence.replace('that\'s', 'that is').replace('wasn\'t', 'was not')

    sentence = sentence.replace('how\'s', 'how is')

    sentence = sentence.replace('you\'ve', 'you have').replace('you\'re', 'you are')

    sentence = sentence.replace('i\'ve', 'i have').replace('they\'re', 'they are')

    sentence = sentence.replace('i\'ll', 'i will').replace('they\'ve', 'they have')

    sentence = sentence.replace('we\'re', 'we are').replace('you\'ll', 'you will')

    sentence = sentence.replace('we\'re', 'we are').replace('we\'ve', 'we have')

    sentence = sentence.replace('we\'ll', 'we will').replace('it\'ll', 'it will').replace('they\'ll', 'they will')

    sentence = sentence.replace('who\'ll', 'who will').replace('who\'ve', 'who have')

    sentence = sentence.replace('he\'ll', 'he will').replace('that\'ll', 'that will')

    sentence = sentence.replace('does\'nt', 'does not').replace('could\'ve', 'could have')

    sentence = sentence.replace('would\'ve', 'would have').replace('what\'re', 'what are')

    sentence = sentence.replace('i\'am', 'i am').replace('who\'re', 'who are')

    sentence = sentence.replace('should\'ve', 'should have').replace('did\'nt', 'did not')

    sentence = sentence.replace('hold\'em', 'hold them').replace('there\'re', 'there are')

    sentence = sentence.replace('do\'nt', 'do not').replace('could\'nt', 'could not')

    return sentence
def find_unigrams(question):

    question = clean_text(question)

    question = replace(question)

    

    word_tokens = question.split(' ')

    word_tokens = [w for w in word_tokens if not w  in stops]

    word_tokens = [w for w in word_tokens if not w == '']

    return word_tokens
def shared_words_in_q2(row):

    q1_tokens = row['q1_tokens']

    q2_tokens = row['q2_tokens']

    

    matching_words = [w for w in q2_tokens if w in q1_tokens]

    return len(matching_words) / (len(q1_tokens) + len(q2_tokens))
def shared_words_in_q1(row):

    q1_tokens = row['q1_tokens']

    q2_tokens = row['q2_tokens']

    matching_words = [w for w in q1_tokens if w in q2_tokens]

    

    return len(matching_words) / (len(q1_tokens) + len(q2_tokens))
train_set['q1_tokens'] = train_set['question1'].map(find_unigrams)

train_set['q2_tokens'] = train_set['question2'].map(find_unigrams)



train_set['q1_length'] = train_set['q1_tokens'].apply(len)

train_set['q2_length'] = train_set['q2_tokens'].apply(len)

train_set['len_diff'] = train_set.apply(lambda x: np.abs(x['q1_length'] - x['q2_length']), axis=1)



train_set['shared_words_q1'] = train_set.apply(lambda x: shared_words_in_q1(x), axis=1)

train_set['shared_words_q2'] = train_set.apply(lambda x: shared_words_in_q2(x), axis=1)
gb_qid  = train_set.groupby('qid1').filter(lambda x: len(x) > 1).groupby('qid1')

duplicate_qid1 = sorted(list(gb_qid.groups))
stats = gb_qid['is_duplicate'].agg({np.sum, np.size})

only_duplicates = stats.loc[stats['sum'] == stats['size']].sort_values(['size'], ascending=False)

duplicate_df = train_set.loc[train_set['qid1'].isin(only_duplicates.index)]
train_set.loc[train_set['qid1'].isin(duplicate_qid1), 'graph_root'] = 1

train_set['graph_root'].fillna(0, inplace=True)

train_set['graph_root'] = train_set['graph_root'].astype(int)
for node in only_duplicates.index:

    group = train_set.loc[train_set['qid1'] == node]

    group1 = train_set.loc[train_set['qid1'].isin(group['qid2'])]

    

    if len(group1) > 0:

        train_set.loc[train_set['qid1'] == node, 'neighbors'] = len(group1)

        

train_set['neighbors'].fillna(0, inplace=True)

train_set['neighbors'] = train_set['neighbors'].astype(int)
clf = RandomForestClassifier()

train_features = ['len_diff', 'shared_words_q1']



def train_data(clf, train_features):

    X = train_set[train_features]

    y = train_set[target]



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    clf = clf.fit(X_train, y_train)



    y_proba = clf.predict_proba(X_test)

    log_loss_score = log_loss(y_test, y_proba)

    metrics.append(log_loss_score)

    return clf
metrics = []

for i in np.arange(5):

    train_data(clf, train_features)
metrics
def preprocess():

    test_set['q1_tokens'] = test_set['question1'].map(find_unigrams)

    test_set['q2_tokens'] = test_set['question2'].map(find_unigrams)



    test_set['q1_length'] = test_set['q1_tokens'].apply(len)

    test_set['q2_length'] = test_set['q2_tokens'].apply(len)

    test_set['len_diff'] = test_set.apply(lambda x: np.abs(x['q1_length'] - x['q2_length']), axis=1)



    test_set['shared_words_q1'] = test_set.apply(lambda x: shared_words_in_q1(x), axis=1)

    test_set['shared_words_q2'] = test_set.apply(lambda x: shared_words_in_q2(x), axis=1)

    return test_set
test_set = preprocess()
def generate_predictions(train_features):

    test_ids = test_set['test_id']

    predictions = clf.predict_proba(test_set[train_features])



    submission = pd.DataFrame(test_ids)



    prediction_set = []

    for i in range(len(predictions)):

        prediction_set.append(predictions[i][1])

    

    prediction_set = pd.DataFrame(prediction_set, columns=[target])

    submission = pd.concat([submission, prediction_set], axis=1)

    return submission
submission = generate_predictions(['len_diff', 'shared_words_q1'])
print(set(pd.isnull(submission[target])))

submission.to_csv("submission.csv", index=False)