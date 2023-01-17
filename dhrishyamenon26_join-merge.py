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
import pandas as pd
from IPython.display import display
from IPython.display import Image
raw_data = {

        'subject_id': ['1', '2', '3', '4', '5'],

        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 

        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
df_a
raw_data = {

        'subject_id': ['4', '5', '6', '7', '8'],

        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 

        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
df_braw_data = {

        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],

        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
df_n = pd.DataFrame(raw_data, columns = ['subject_id','test_id'])
df_n
df_new = pd.concat([df_a, df_b])
df_new
pd.concat([df_a, df_b], axis=1)
pd.merge(df_new, df_n, on='subject_id')
pd.merge(df_new, df_n, left_on='subject_id', right_on='subject_id')
pd.merge(df_a, df_b, on='subject_id', how='outer')
pd.merge(df_a, df_b, on='subject_id', how='inner')
pd.merge(df_a, df_b, on='subject_id', how='right')
pd.merge(df_a, df_b, on='subject_id', how='left')
pd.merge(df_a, df_b, on='subject_id', how='left', suffixes=('_left', '_right'))
pd.merge(df_a, df_b, right_index=True, left_index=True)