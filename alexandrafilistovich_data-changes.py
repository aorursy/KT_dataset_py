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
import pandas as pd

df = pd.read_csv('/kaggle/input/mydata/Test.csv', sep=',')

df_tr = pd.read_csv('/kaggle/input/mydata/Test.csv', sep=',')

df_tr.head()
df_ans = df_tr[['TransactionId']]

df_ans['IsDefaulted'] = df_tr['CurrencyCode']

df_ans.head()
dfa = df_ans



for i in range(dfa.shape[0]):

    if df_tr.at[i,'TransactionStatus'] == 1:

        dfa.at[i,'IsDefaulted'] = -1

    else:

        dfa.at[i,'IsDefaulted'] = -1

dfa.head(20)
dfa.to_csv('dfa.csv',index=False)