import numpy as np

import pandas as pd 

import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

import re

from string import punctuation

import seaborn as sns
train = pd.read_csv("../input/train.csv")[:1000]

test = pd.read_csv("../input/test.csv")[:1000]
train = train.fillna("")
print('Any null in training set?')

print(train.any().isnull())

print()

print('Any null in the testing set?')

print(test.any().isnull())
print(train.head())
a = 0



for i in range(a,a+10):

    print(train.question1[i])

    print(train.question2[i])

    print(train.is_duplicate[i])

    print()
def Question_Word(questions):

    Q_words = []

    for q in questions:

        j = 0

        if 'What' in q: j = 1

        elif 'what' in q: j = 1

        elif 'Which' in q: j = 2

        elif 'How' in q: j = 3

        elif 'Can' in q: j = 4

        elif 'When' in q: j = 5

        elif 'Why' in q: j = 6

        elif 'Should' in q: j = 7

        elif 'Does' in q: j = 8

        Q_words.append(j)

    return(Q_words) 
train['qWord1'] = Question_Word(train.question1)

train['qWord2'] = Question_Word(train.question2)

train['qWordMatch'] = np.where(train.qWord1 == train.qWord2,1,0)
print(train.qWordMatch.value_counts())

print(sns.barplot(x = train.qWordMatch, y = train.is_duplicate))
train['q1len'] = train['question1'].str.len()

train['q2len'] = train['question2'].str.len()



train['q1_n_words'] = train['question1'].apply(lambda row: len(row.split(" ")))

train['q2_n_words'] = train['question2'].apply(lambda row: len(row.split(" ")))



def normalized_word_share(row):

    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))

    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    

    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))





train['word_share'] = train.apply(normalized_word_share, axis=1)
sns.distplot(train.loc[train.is_duplicate == 1,'word_share'], color = 'red')

sns.distplot(train.loc[train.is_duplicate == 0,'word_share'], color = 'blue')
q_words = ['what','which','how','why','should','does','when','can','i']



def proper_check(q):

    proper = []

    for word in q.split(" "):

        if word == '': continue

        elif (word[0].isupper()) & (word.lower() not in q_words): proper.append(word)

    return(proper)



def proper_word_share(row):

    s1 = set(proper_check(row['question1']))

    s2 = set(proper_check(row['question2']))

    if (len(s1) + len(s2)) == 0: return(-1)

    else: return 1.0 * len(s1 & s2)/(len(s1) + len(s2))



train['proper_share'] = train.apply(proper_word_share, axis = 1)
sns.distplot(train.loc[(train.is_duplicate == 1) & (train.proper_share != -1),'proper_share'], color = 'red')

sns.distplot(train.loc[(train.is_duplicate == 0) & (train.proper_share != -1),'proper_share'], color = 'blue')
print('% of proper nouns shared between non duplicate questions:')

print(train.loc[(train.is_duplicate == 1) & (train.proper_share != -1),'proper_share'].mean())

print()

print('% of proper nouns shared between duplicate questions:')

print(train.loc[(train.is_duplicate == 0) & (train.proper_share != -1),'proper_share'].mean())
sns.distplot(train.loc[train.is_duplicate == 1,'q1_n_words'], color = 'red')

sns.distplot(train.loc[train.is_duplicate == 0,'q2_n_words'], color = 'blue')
train['len_ratio'] = train.q1_n_words * 1.0 / train.q2_n_words



sns.distplot(train.loc[train.is_duplicate == 1,'len_ratio'], color = 'red')

sns.distplot(train.loc[train.is_duplicate == 0,'len_ratio'], color = 'blue')
dfq1, dfq2 = train[['qid1','question1']], train[['qid2','question2']]

dfq1.columns = ['qid1','question']

dfq2.columns = ['qid2','question']



# merge question dfs and fill na

dfqa = pd.concat((dfq1,dfq2),axis = 0).fillna("")

print(dfqa.head())

from sklearn.feature_extraction.text import TfidfVectorizer



tfidf1 = TfidfVectorizer(max_features = 300, stop_words= 'english')



mq1 = tfidf1.fit_transform(dfqa['question'].values)

tfidf_diff = np.abs(mq1[:dfq1.shape[0]] - mq1[dfq1.shape[0]:])
print(dict(zip(tfidf1.get_feature_names(),tfidf1.idf_)))
print(dfqa.question[0].values,mq1[0])