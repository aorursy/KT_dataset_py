# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt #Library for general visualizations

import seaborn as sns #For more beautiful visualizations

import numpy as np #Library that handles mathematical operations

import pandas as pd #Working with .csv files

import time #General Python time library



#Magic command to the jupyter notebook that we want all visualizations to stay within the file

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
test_identity = pd.read_csv("../input/test_identity.csv")

test_transaction = pd.read_csv("../input/test_transaction.csv")

train_transaction = pd.read_csv("../input/train_transaction.csv")

train_identity = pd.read_csv("../input/train_identity.csv")
train_transaction.head()
train_transaction.shape
targets = train_transaction.isFraud

ax = sns.countplot(targets, label="Count", palette="Set3")

Y,N = targets.value_counts()

print('Is Fraud:',Y,"Percent:",int(Y/(Y+N)*100),"%")

print('Is not fraud:',N,"   Percent:",int(N/(Y+N)*100),"%")
for col in train_transaction.columns:

    print (col,": ",(train_transaction[col].isna().sum())/590540*100)

    