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
df1 = pd.read_csv('../input/sctp-blend-data/submission_1.csv')

df2 = pd.read_csv('../input/sctp-blend-data/submission_2.csv')

df3 = pd.read_csv('../input/sctp-blend-data/submission_3.csv')

df4 = pd.read_csv('../input/sctp-blend-data/submission_4.csv')
blend = df4['target'] *0.25 + df3['target'] * 0.25 + df2['target'] * 0.25 + df1['target'] * 0.25 

sample = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
sample['target'] = blend

sample.to_csv('blend_ver10.csv',index=False)