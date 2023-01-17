# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sub = pd.read_csv('/kaggle/input/my-sub/mysub.csv').fillna('aaa')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv').fillna('')

# sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

# sub = pd.read_csv('/kaggle/input/submission/tstsub.csv')
for i in range(sub.shape[0]):

    if (str(type(sub.iloc[i, 0])) != "<class 'str'>") or (str(type(sub.iloc[i, 1])) != "<class 'str'>"):

        print('class!!!!')

        

    if (str(type(sub.iloc[i, 0])) in '"') or (str(type(sub.iloc[i, 1])) in '"'):

        print('quate!!!!')

print('done')
sub.info()
sub.head()
all = []

for k in range(sub.shape[0]):

    text1 = " "+" ".join(sub.loc[k,'selected_text'].split())

#     enc = tokenizer.encode(text1)

#     st = tokenizer.decode(enc.ids[a-2:b-1])

    all.append(text1)
all[0: 5]
# len(all)
# sample.shape
test['selected_text'] = all

submission=test[['textID','selected_text']]

submission.to_csv('submission.csv',index=False)
submission.head(5)
# test['selected_text'] = all

# test[['textID','selected_text']].to_csv('/kaggle/working/submission.csv', index=False)
# f = open('submission.csv','w')

# f.write('textID,selected_text\n')

# for index, row in sub.iterrows():

#     f.write('%s,"%s"\n'%(row.textID,row.selected_text))

# f.close()
# sam = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
# submission.info()
# sam.info()
# y = []

# for i in submission.selected_text:

# #     print(i)

#     y.append('"' + i + '"') 

# submission.drop(columns=["selected_text"])

# submission['selected_text'] = y

# submission
# submission.to_csv('submission.csv', quoting = csv.QUOTE_ALL, index = False)
# submission.head()