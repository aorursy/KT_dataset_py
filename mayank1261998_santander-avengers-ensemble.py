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
df=pd.read_csv('../input/modified-naive-bayes-santander-0-899/submission.csv')

df3=pd.read_csv('../input/santander-customer-transaction-eda/submission_baseline_forest.csv')

df4=pd.read_csv('../input/easy-nn-for-santander/subm_nn_bl_14_2019-03-25_1.csv')

df5=pd.read_csv('../input/easy-nn-for-santander/subm_nn_bl_14_2019-03-25_2.csv')
ensemble=(df.target.values+df3.target.values+df4.target.values+df5.target.values)/4
df.target.describe()
ensemble.mean()
df['target']=ensemble
df.to_csv('Ensembled_sub.csv',index=False)