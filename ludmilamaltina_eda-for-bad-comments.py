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
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib_venn import venn2

import re
train = pd.read_csv('../input/train.csv')

train.shape
train.dtypes
train.isnull().sum()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

features = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



for i, f in enumerate(features):

    sns.countplot(x=f, data=train, ax=axes[i // 3, i % 3])
def venn_diagram2(sets, labels):

    '''функция для построения диаграмм Венна с двумя множествами'''

    c = venn2(subsets=sets, set_labels=labels)

    c.get_patch_by_id('01').set_color('#FF0000')  # red

    c.get_patch_by_id('10').set_color('#FFFF00')  # yellow

    c.get_patch_by_id('11').set_color('#FF4500')  # orange

    

    for i in ('01', '10', '11'):

        c.get_patch_by_id(i).set_alpha(0.6)
def values_and_diagram2(f1, f2):

    '''функция для печати количества элементов в двух множествах и построения для них диаграммы Венна'''

    f1_and_not_f2 = train.loc[(train[f1] == 1) & (train[f2] == 0)].shape[0]

    f2_and_not_f1 = train.loc[(train[f1] == 0) & (train[f2] == 1)].shape[0]

    f1_and_f2 = train.loc[(train[f1] == 1) & (train[f2] == 1)].shape[0]

    

    print('{}: {}'.format(f1, train.loc[(train[f1] == 1)].shape[0]))

    print('{}: {}'.format(f2, train.loc[(train[f2] == 1)].shape[0]))

    print('{} and not {}: {}'.format(f1, f2, f1_and_not_f2))

    print('{} and not {}: {}'.format(f2, f1, f2_and_not_f1))

    print('{} and {}: {}'.format(f1, f2, f1_and_f2))

    

    venn_diagram2((f1_and_not_f2, f2_and_not_f1, f1_and_f2), (f1, f2))
values_and_diagram2('toxic', 'severe_toxic')
values_and_diagram2('toxic', 'obscene')
# количество типов токсичности, использующихся одновременно

# 0 - все значения столбцов toxic, severe_toxic, obscene, threat, insult, identity_hate нулевые, то есть комментарий хороший

# 6 - все значения этих столбцов единичные, то есть комментарий обладает всеми типами токсичности

train['toxic_score'] = train['toxic'] + train['severe_toxic'] + train['obscene'] + train['threat'] + train['insult'] + train['identity_hate'] 

train['toxic_score'].value_counts()
plt.figure(figsize=(7, 5))

sns.countplot(x='toxic_score', data=train)
train['count_words'] = train['comment_text'].map(lambda x: len(re.findall('[A-Za-z0-9-*]+', x)))  # слова содержат буквы английского алфавита, цифры, дефисы и звёздочки

train['count_words'].describe()
plt.hist(train['count_words'], bins=50)

plt.show()
train['count_words'].groupby(train['toxic']).describe()
train['count_unique_words'] = train['comment_text'].map(lambda x: len(set(re.findall('[A-Za-z0-9-*]+', x))))  # количество уникальных слов

train['part_unique_words'] = train['count_unique_words']/train['count_words']  # доля уникальных слов

train['part_unique_words'] = train['part_unique_words'].fillna(0)  # если было деление на 0, то получилось NaN, заменим его на 0 
for feature in ('count_unique_words', 'part_unique_words'):

    print(feature)

    print(train[feature].groupby(train['toxic']).describe(), '\n\n')
train['count_urls'] = train['comment_text'].map(lambda x: len(re.findall('https?://[\S]+', x)))  # количество ссылок
train['count_urls'].groupby(train['toxic']).describe()
# количество восклицательных знаков

train['exclamations'] = train['comment_text'].map(lambda x: x.count('!')) 



# количество символов пунктуации .?!

train['punctuation'] = train['comment_text'].map(lambda x: x.count('!') + x.count('?') + x.count('.'))



train['part_exclamations'] = train['exclamations']/train['punctuation']  # доля восклицательных знаков среди символов пунктуации .?!

train['part_exclamations'] = train['part_exclamations'].fillna(0)  # если было деление на 0, то получилось NaN, заменим его на 0 



# количество повторяющихся восклицательных знаков

train['exclamations_repeated'] = train['comment_text'].map(lambda x: len(re.findall('!{2,}', x)))  



train['part_exclamations_repeated'] = train['exclamations_repeated']/train['punctuation']  # доля повторящющихся восклицательных знаков среди символов пунктуации .?! 

train['part_exclamations_repeated'] = train['part_exclamations_repeated'].fillna(0)  # если было деление на 0, то получилось NaN, заменим его на 0 
for feature in ('exclamations', 'part_exclamations', 'exclamations_repeated', 'part_exclamations_repeated'):

    print(feature)

    print(train[feature].groupby(train['toxic']).describe(), '\n\n')
# слова написаны большими латинскими буквами и могут содержать * и состоят из трёх и более букв (это делается, чтобы не учитывать сокращения, например, UK)

train['caps_words'] = train['comment_text'].map(lambda x: len(re.findall('[A-Z0-9-*]{3,}', x)))  



train['part_caps_words'] = train['caps_words']/train['count_words']  # доля слов, написанных заглавными буквами

train['part_caps_words'] = train['part_caps_words'].fillna(0)  # если было деление на 0, то получилось NaN, заменим его на 0 
for feature in ('caps_words', 'part_caps_words'):

    print(feature)

    print(train[feature].groupby(train['toxic']).describe(), '\n\n')
train['stars'] = train['comment_text'].map(lambda x: x.count('*')/len(x))  # доля звёздочек среди символов текста

print(train['stars'].groupby(train['obscene']).describe())
def plot_heatmap(figsize, columns):

    plt.figure(figsize=figsize)

    corr_matrix = train[columns].corr()

    sns.heatmap(corr_matrix, annot=True, cbar=False)

    plt.yticks(size=12, rotation=0) 

    plt.xticks(size=12, rotation=90) 

    plt.show()
plot_heatmap((14,14), ['count_words', 'count_unique_words', 'part_unique_words', 'count_urls',

                     'exclamations', 'exclamations_repeated', 'part_exclamations', 'part_exclamations_repeated',  

                     'caps_words', 'part_caps_words', 'stars',

                     'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
train.drop(columns=['count_words', 'part_unique_words', 

                    'punctuation', 'exclamations', 'exclamations_repeated', 'part_exclamations_repeated',

                    'caps_words', 'stars', 'toxic_score'], inplace=True)
train.columns
plot_heatmap((8, 8), ['count_unique_words', 'count_urls', 'part_exclamations', 'part_caps_words',

                     'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])