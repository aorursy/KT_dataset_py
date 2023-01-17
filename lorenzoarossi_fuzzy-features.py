# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from difflib import SequenceMatcher

from fuzzywuzzy import fuzz



from nltk.corpus import stopwords

from nltk import word_tokenize

stop_words = stopwords.words('english')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
clock_start_time = time.process_time()

data = pd.read_csv("../input/"+"train.csv", header=0) 

data = data.drop(['id', 'qid1', 'qid2'], axis=1)

print("loaded training data")





data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)



print("testing features computed in", "{0:.4f}".format(time.process_time()-clock_start_time), "seconds")



data.drop(['question1', 'question2', 'is_duplicate'], axis=1).to_csv('f_Xtr.csv', index=False)

print ('fuzzy features saved!')
clock_start_time = time.process_time()

BLOCK_LEN = 586449

k=3

row_range = range(1,k*BLOCK_LEN+1*int(k!=0))

data = pd.read_csv("../input/"+"test.csv", nrows=BLOCK_LEN,  skiprows=row_range) 

data = data.drop(['test_id'], axis=1)

print("loaded testing data")



data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)

print(3)

data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

print(5)

data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)



print("testing fuzzy features computed in", "{0:.4f}".format(time.process_time()-clock_start_time), "seconds")



data.drop(['question1', 'question2'], axis=1).to_csv('f_Xts_a3.csv', index=False)

print ('fuzzy features saved for testing data!')