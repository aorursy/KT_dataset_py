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
df_test = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

df_sample = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')

df_train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

df_train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

df_test = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

df_test.shape , df_sample.shape , df_train_identity.shape ,df_train_transaction.shape ,df_test.shape
df_test.head(2)
df_train_identity.columns
df_train_transaction.columns
df_tt = pd.merge(df_train_transaction,df_train_identity,how='left')
df_tt.shape
df_tt.columns
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(30,15))

sns.heatmap(df_tt.corr(),annot=True)