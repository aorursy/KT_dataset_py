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
train=pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

train_id=pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

test=pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

test_id=pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

sample=pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
train.head()
train_id.head()
train[['ProductCD','card1','card2','card3','card4','card5','card6','addr1', 'addr2','P_emaildomain',

'R_emaildomain','M1','M9']]
j=0

empty=[]

for i in train.isna().sum():

    j=j+1

    if(i>.6*len(train)):

        print(j,i)

        empty.append(j)
len(train)