# import library 

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)  

pd.set_option('display.float_format', '{:20,.2f}'.format)

import matplotlib.pyplot as plt

plt.figure(figsize=(16,4))

import seaborn as sns

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler
# read data 

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
train.head(3)
# IDENTITY_COLUMNS (신원을 파악할 수 있는 변수)

IDENTITY_COLUMNS = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish','muslim', 'black', 'white', 'psychiatric_or_mental_illness']



# AUX_COLUMNS (additional toxicity subtype attributes) 

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
tmp = train.isnull().sum(axis=0) / len(train)

tmp[tmp > 0]
identities = [

    'male','female','transgender','other_gender','heterosexual','homosexual_gay_or_lesbian',

    'bisexual','other_sexual_orientation','christian','jewish','muslim','hindu','buddhist',

    'atheist','other_religion','black','white','asian','latino','other_race_or_ethnicity',

    'physical_disability','intellectual_or_learning_disability','psychiatric_or_mental_illness','other_disability'

]



tmp = train.loc[:, ['target'] + identities ].dropna()

toxic_df = tmp[tmp['target'] >= .5][identities]

non_toxic_df = tmp[tmp['target'] < .5][identities]

print('toxic_df',len(toxic_df)/len(tmp))

print('non_toxic_df',len(non_toxic_df)/len(tmp))
print(train[train.target==1].iloc[1,2])
print(train[train.severe_toxicity==1].iloc[0,2])
print(train[train.obscene==1].iloc[1,2])
print(train[train.identity_attack==1].iloc[1,2])
print(train[train.insult==1].iloc[1,2])
print(train[train.threat==1].iloc[1,2])
# tox label counts

tox_labels = train[AUX_COLUMNS].copy()

rowsums = tox_labels.sum(axis=1)

train['clean']=(rowsums==0)

tox_labels['clean']=train['clean'].copy()

x = tox_labels.sum().copy()

sns.barplot(x.index, x.values)
# corr matrix

corr = train[AUX_COLUMNS].corr()

print(corr)

sns.heatmap(corr) 
train['total_length'] = train['comment_text'].apply(len)

train['capitals'] = train['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

train['caps_vs_length'] = train.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)

train['num_exclamation_marks'] = train['comment_text'].apply(lambda comment: comment.count('!'))

train['num_question_marks'] = train['comment_text'].apply(lambda comment: comment.count('?'))

train['num_punctuation'] = train['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))

train['num_symbols'] = train['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))

train['num_words'] = train['comment_text'].apply(lambda comment: len(comment.split()))

train['num_unique_words'] = train['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))

train['words_vs_unique'] = train['num_unique_words'] / train['num_words']

train['num_smilies'] = train['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
train['all_capitals_YN'] = train.apply(lambda row: float(row['capitals'])==float(row['total_length']),axis=1)
features = [

'total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks','num_question_marks', 

'num_punctuation', 'num_words', 'num_unique_words','words_vs_unique', 'num_smilies', 'num_symbols',

'all_capitals_YN'

]    

tmp = train[['target']+features].copy()

tmp = tmp.apply(lambda x:np.where(x>=0.5,1,0))

tmp.groupby(features)['target'].agg([np.mean,np.sum,np.size]).T
pd.crosstab(tmp.target,tmp.all_capitals_YN)