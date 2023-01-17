# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import time

from datetime import datetime

from collections import Counter





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input", "-la"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')

members = pd.read_csv('../input/members.csv')

transactions = pd.read_csv('../input/transactions.csv')

# user_logs = pd.read_csv('../input/user_logs.csv')

sample_submission_zero = pd.read_csv('../input/sample_submission_zero.csv')
sample_submission_zero.head()

sample_submission_zero['is_churn'] = 1

sample_submission_zero.head()

sample_submission_zero.to_csv('Test.csv', index=False)
print(check_output(["ls", "-la"]).decode("utf8"))

train.head(20)
members.shape

# len(members['msno'].unique())

members.head(100)
transactions.head(20)