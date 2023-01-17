import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sub_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sub_Priyanshu = pd.read_csv("/kaggle/input/bert-fastai/submission.csv")



train.head()
train = train.fillna('None')

ag = train.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(columns={'text':'Count', 'target':'Disaster Probability'})

ag.sort_values('Disaster Probability', ascending=False).head(10)
keyword_list = list(ag[(ag['Count']>2) & (ag['Disaster Probability']>=0.9)].index)

keyword_list
ids = test['id'][test.keyword.isin(keyword_list)].values

sub_Priyanshu['target'][sub_Priyanshu['id'].isin(ids)] = 1

sub_Priyanshu.head()
sub_Priyanshu.to_csv('sub_Priyanshu_modified.csv', index=False)