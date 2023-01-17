# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

import time

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pd.read_csv('../input/sample_submission.csv')
train_df_original = pd.read_csv('../input/train.tsv', delimiter='\t')

train_df_original
df = pd.DataFrame()



start_time = time.time()

for key, row in train_df_original.iterrows():

    if type(row.category_name) != 'float':

        cat_split = row.category_name.split('/')

    else:

        cat_split = [np.nan] * 3

    cat_series = pd.Series(cat_split, index = ['category1', 'category2', 'category3'], name=key)        

    processed_row = row.append(cat_series)

    df = df.append(processed_row)

    if key % 10000 == 0:

        elapsed_time = time.time() - start_time

        logging.debug('{}/{} rows have been processed. {} sec'. format(key, len(train_df_original), round(elapsed_time, 2)))

    break # for debug    

df
df.category_name