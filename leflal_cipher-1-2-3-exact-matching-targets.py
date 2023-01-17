import numpy as np 
import pandas as pd 

import os

from IPython.core.display import display
from sklearn.datasets import fetch_20newsgroups
competition_path = '20-newsgroups-ciphertext-challenge'
test = pd.read_csv('../input/' + competition_path + '/test.csv').rename(columns={'ciphertext' : 'text'})
train_p = fetch_20newsgroups(subset='train')
test_p = fetch_20newsgroups(subset='test')
df_p = pd.concat([pd.DataFrame(data = np.c_[train_p['data'], train_p['target']],
                                   columns= ['text','target']),
                      pd.DataFrame(data = np.c_[test_p['data'], test_p['target']],
                                   columns= ['text','target'])],
                     axis=0).reset_index(drop=True)
df_p['target'] = df_p['target'].astype(np.int8)
def find_targets(p_indexes_set):
    return np.sort(df_p.loc[p_indexes_set]['target'].unique())
pickle_1_path = '../input/test-1/'

df_p_indexes_1 = pd.read_pickle(pickle_1_path + 'df_p_indexes-1.pkl')
df_p_indexes_1['target'] = df_p_indexes_1['p_indexes'].map(find_targets)
display(df_p_indexes_1[df_p_indexes_1['target'].map(len) > 1 ])
test = test.join(df_p_indexes_1[['target']])
pickle_2_path = '../input/test-2/'

df_p_indexes_2 = pd.read_pickle(pickle_2_path + 'df_p_indexes-2.pkl')
df_p_indexes_2['target'] = df_p_indexes_2['p_indexes'].map(find_targets)
display(df_p_indexes_2[df_p_indexes_2['target'].map(len) > 1 ])
test.loc[df_p_indexes_2.index,'target'] = df_p_indexes_2['target']
pickle_3_path = '../input/cipher-3-solution/'

df_p_indexes_3 = pd.read_pickle(pickle_3_path + 'test_3.pkl')
df_p_indexes_3['target'] = df_p_indexes_3['p_indexes'].map(find_targets)
display(df_p_indexes_3[df_p_indexes_3['target'].map(len) > 1 ])
test.loc[df_p_indexes_3.index,'target'] = df_p_indexes_3['target']
test.head()
test[test['target'].isnull()]['difficulty'].unique()
test.to_pickle('test_123.pkl')
test.loc[test['difficulty'] < 4,'target'] = test.loc[test['difficulty'] < 4,'target'].map(lambda x: x[0])
#You can implement the target choice you want within the possible targets here
test.to_pickle('test_sub.pkl')