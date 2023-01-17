# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tqdm import tqdm

import os

import datetime

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
submission_prob = pd.read_csv('../input/lightgbm-with-usertag/prob_submsission.csv')

submission_tag = pd.read_csv('../input/lightgbm-with-usertag/submsission.csv')
submission_tag.head()
#统计第一大类的概率

prob = []

for i in range(len(submission_tag)):

    tag = int(submission_tag['0'][i])

    prob.append(submission_prob[str(tag)][i])

submission_tag['prob'] = prob

submission_tag['prob'].hist()
test_data = pd.read_csv('../input/finance/data/data/test.csv')

test_data['due_date'] = pd.to_datetime(test_data['due_date']) 

test_data.head()
test_data[test_data['listing_id']==5483324]
((pd.to_datetime(test_data['due_date']) - pd.to_datetime(test_data['auditing_date'])).dt.days).hist()
month_days1 = {3:29,4:32}

month_days2 = {3:29,4:31}

month_array = pd.to_datetime(test_data['due_date']).dt.month

day_array = pd.to_datetime(test_data['auditing_date']).dt.day

j = 1

i =1

int(submission_prob.iloc[j].sort_values(ascending=False).index[i+1]) <= month_days1[month_array[1]]
#根据prob决定每个list的提交个数

prob = []

index = []

for j in tqdm(range(len(submission_prob))):

    prob_try = []

    index_try = []

    if submission_prob.iloc[j].sort_values(ascending=False).values[1]>0.6:

        length = 1

    else: 

        if submission_prob.iloc[j].sort_values(ascending=False).values[1]>0.3:

            length = 3

        else:

            if submission_prob.iloc[j].sort_values(ascending=False).values[1]>0.1:

                length = 5

            else:

                length = 10

    month_tag = month_array[j]

    day_tag = day_array[j]

    if day_tag == 31:

        month_days = month_days2

    else:

        month_days = month_days2

    for i in range(length):

        if int(submission_prob.iloc[j].sort_values(ascending=False).index[i+1]) <= month_days[month_tag]:

            prob_try.append(submission_prob.iloc[j].sort_values(ascending=False).values[i+1])

            index_try.append(submission_prob.iloc[j].sort_values(ascending=False).index[i+1])

        else:

            print(submission_prob.iloc[j].sort_values(ascending=False).index[i+1])

            print(month_days[month_tag])

    prob.append(prob_try)

    index.append(index_try)
#将概率归一化

prob_ = []

for p in tqdm(prob):

    p = np.array(p)/sum(p)

    prob_.append(p)
submission = pd.DataFrame(columns=['listing_id','repay_amt','repay_date'])

submission.head()
num = -1

for ind in tqdm(range(len(index))):

    index_try = index[ind]

    prob_try = prob_[ind]

    for i,j in zip(index_try,prob_try):

        num += 1

        if int(i) == 0:

            continue

        else:

            listing_id = int(test_data['listing_id'][ind])

            repay_amt = test_data['due_amt'][ind] * j

            repay_date = test_data['due_date'][ind] - datetime.timedelta(days=int(i)-1)

            submission.at[num] = [listing_id,repay_amt,repay_date]



        
submission['repay_date'] = pd.to_datetime(submission['repay_date']).apply(lambda x:x.strftime('%Y/%m/%d'))
submission.head()
submission.to_csv('submission.csv',index=False)