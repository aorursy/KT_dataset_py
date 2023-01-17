import pyLDAvis.gensim
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
train = pd.read_csv('../input/en_train.csv')
test = pd.read_csv('../input/en_test.csv')
train.tail()
test.tail()
train['class'].unique()
train['class'].value_counts().sort_values(ascending = False)
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(y = train['class'],data = train)

train[train['class']=='PUNCT'].head()
train[train['class']=='DATE'].head()
train[train['class']=='LETTERS'].head()
train[train['class']=='CARDINAL'].head()
train[train['class']=='VERBATIM'].head()
train[train['class']=='DECIMAL'].head()
train[train['class']=='MEASURE'].head()
train[train['class']=='MONEY'].head()
train[train['class']=='ORDINAL'].head()
train[train['class']=='TIME'].head()
sample_submission = pd.read_csv("../input/en_sample_submission.csv")
sample_submission.head()
train['sentences'] = train.groupby("sentence_id")["sentence_id"].count()
train['sentences'].describe()
test['sentences'] = test.groupby("sentence_id")["sentence_id"].count()
test['sentences'].describe()
plt.figure(figsize=(30,8))
sns.countplot(x = train['sentences'],data = train)
plt.xticks(rotation = 90)
plt.show()
train[train['sentences'] ==256]
train[train['sentence_id']==520453]
max_sentance = train[train['sentence_id']==520453].before.values.tolist()
max_sentance = ' '.join(max_sentance)
max_sentance
train[train['sentences']==2]
min_sentance = train[train['sentence_id']==41].before.values.tolist()
min_sentance = ' '.join(min_sentance)
min_sentance #you can replace the above number to get different sentances which have 2 words only.
