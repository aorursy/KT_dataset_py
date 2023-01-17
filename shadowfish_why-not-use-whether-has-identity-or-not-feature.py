# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/ieee-fraud-detection'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# General imports

import os, sys, gc, warnings, random

import numpy as np

import pandas as pd

from sklearn import metrics

from sklearn.model_selection import train_test_split, KFold

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

warnings.filterwarnings('ignore')
# load data

train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

train_identity =  pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")



test_identity =  pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

test_transaction =  pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
# add a new feature to represent one transaction has identity or not

train_transaction["has_identity"] = np.where(train_transaction["TransactionID"].isin(train_identity["TransactionID"].unique()),1,0)

test_transaction["has_identity"] = np.where(test_transaction["TransactionID"].isin(test_identity["TransactionID"].unique()),1,0)
train_transaction.groupby(["has_identity","isFraud"]).count()["TransactionID"].rename("freq").reset_index()