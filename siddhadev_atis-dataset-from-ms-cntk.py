# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/atis"))





# Any results you write to the current directory are saved as output.
import pickle



DATA_DIR="../input/atis"



def load_ds(fname='atis.train.pkl'):

    with open(fname, 'rb') as stream:

        ds,dicts = pickle.load(stream)

    print('Done  loading: ', fname)

    print('      samples: {:4d}'.format(len(ds['query'])))

    print('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))

    print('   slot count: {:4d}'.format(len(dicts['slot_ids'])))

    print(' intent count: {:4d}'.format(len(dicts['intent_ids'])))

    return ds,dicts
train_ds, dicts = load_ds(os.path.join(DATA_DIR,'atis.train.pkl'))

test_ds, dicts  = load_ds(os.path.join(DATA_DIR,'atis.test.pkl'))
t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])

i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])

query, slots, intent =  map(train_ds.get, ['query', 'slot_labels', 'intent_labels'])



for i in range(5):

    print('{:4d}:{:>15}: {}'.format(i, i2in[intent[i][0]],

                                    ' '.join(map(i2t.get, query[i]))))

    for j in range(len(query[i])):

        print('{:>33} {:>40}'.format(i2t[query[i][j]],

                                     i2s[slots[i][j]]  ))

    print('*'*74)