# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import spacy

import csv

from itertools import zip_longest



data_train=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv', names=[0,1,2,3], encoding='utf-8')

data_test=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv', names=[0,1,2], encoding='utf-8')

data_sub=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv', names=[0,1], encoding='utf-8')



positive=pd.read_csv('/kaggle/input/sentiment-wordlist/eng_positivewords.csv', names=[0], encoding='utf-8')

negative=pd.read_csv('/kaggle/input/sentiment-wordlist/english_slang.csv', names=[0,1,2], encoding='utf-8')



train_text=list(data_train[1])

train_selected=list(data_train[2])

train_label=list(data_train[3])



test_text=list(data_test[1])

test_label=list(data_test[2])



sub=list(data_sub[0])



pos=list(positive[0])

neg=[]

for i in list(negative[1]):

    neg.append(i)

for i in list(negative[2]):

    neg.append(i)

#print(pos)

#print(neg)



selected_text=[]

for i in range(len(test_text)):

    shortp,shortn=[],[]

    separator=','

    if test_label[i]=='positive':

        for j in test_text[i]:

            if j in pos:

                shortp.append(j)

        selected_text.append(separator.join(shortp))

    if test_label[i]=='negative':

        for j in test_text[i]:

            if j in neg:

                shortn.append(j)

        selected_text.append(separator.join(shortn))

    if test_label[i]=='neutral':

        selected_text.append(test_text[i])

#print(selected_text)

selected_text.insert(0,'selected_text')



d=[sub, selected_text]

export_data = zip_longest(*d, fillvalue = '')



with open("/kaggle/working/submission.csv", "w", encoding="ISO-8859-1", newline='') as f:

    wr = csv.writer(f)

      #wr.writerow(("List1", "List2"))

    wr.writerows(export_data)

f.close()

#f.close()

   #for i in selected_text:

   #f=open('temp.csv', 'a')

      #f.write(i)

      #f.write('\n')

   #f.close()

  # writer.writerows(selected_text)

#df=pd.DataFrame('submission.csv', 'r', columns)




