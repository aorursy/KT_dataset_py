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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
# look at some diplicated question pairs

for i in train[train['is_duplicate'] == 1].index[:10]:

    print('{}.'.format(i))

    print('question1: {}'.format(train[train['is_duplicate'] == 1].loc[i, 'question1']))

    print('question2: {}'.format(train[train['is_duplicate'] == 1].loc[i, 'question2']))

    print('---------------------------------------------------')
# look at some no-diplicated question pairs

for i in train[train['is_duplicate'] == 0].index[:30]:

    print('{}.'.format(i))

    print('question1: {}'.format(train[train['is_duplicate'] == 0].loc[i, 'question1']))

    print('question2: {}'.format(train[train['is_duplicate'] == 0].loc[i, 'question2']))

    print('---------------------------------------------------')
train.info()
test.info()
train.fillna('xxxxx', inplace=True)

test.fillna('xxxxx', inplace=True)
# create columns with both question1 and question2

train['question_pair'] = train['question1'] + '. ' + train['question2']

test['question_pair'] = test['question1'] + '. ' + test['question2']
### Get a set of all Symbols in both train and text
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(analyzer='char')
count_vectorizer.fit(pd.concat([train['question_pair'], test['question_pair']]))