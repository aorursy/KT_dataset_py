# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print('\n'.join(os.listdir("../input")))



# Any results you write to the current directory are saved as output.
def read_tsv(data_file):

    text_data = list()

    labels = list()

    infile = open(data_file, encoding='utf-8')

    for line in infile:

        if not line.strip():

            continue

        label, text = line.split('\t')

        text_data.append(text)

        labels.append(label)

    return text_data, labels



###############################################################



pos_train_file = '../input/train_Arabic_tweets_positive_20190413.tsv'

neg_train_file = '../input/train_Arabic_tweets_negative_20190413.tsv'



pos_test_file = '../input/test_Arabic_tweets_positive_20190413.tsv'

neg_test_file = '../input/test_Arabic_tweets_negative_20190413.tsv'



pos_train_data, pos_train_labels = read_tsv(pos_train_file)

neg_train_data, neg_train_labels = read_tsv(neg_train_file)



pos_test_data, pos_test_labels = read_tsv(pos_test_file)

neg_test_data, neg_test_labels = read_tsv(neg_test_file)
pos = pos_train_data + pos_test_data 

neg = neg_train_data + neg_test_data
! mkdir -p arabic_tweets/pos 

! mkdir -p arabic_tweets/neg
for i in range(len(pos)):

    outfile = open('arabic_tweets/pos/' + str(i) + '.txt', mode='w', 

                   encoding='utf-8')

    outfile.write(pos[i])
for i in range(len(neg)):

    outfile = open('arabic_tweets/neg/' + str(i) + '.txt', mode='w', 

                   encoding='utf-8')

    outfile.write(neg[i])
import shutil

shutil.make_archive('arabic_tweets', 'zip', 'arabic_tweets')
! ls 
! rm -r arabic_tweets  
! ls 