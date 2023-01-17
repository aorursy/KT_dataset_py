# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission_df = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")

specs_df = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

test_df = pd.read_csv("../input/data-science-bowl-2019/test.csv")

train_df = pd.read_csv("../input/data-science-bowl-2019/train.csv")

train_labels_df = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
print(f"train shape: {train_df.shape}")

print(f"test shape: {test_df.shape}")

print(f"train labels shape: {train_labels_df.shape}")

print(f"specs shape: {specs_df.shape}")

print(f"sample submission shape: {sample_submission_df.shape}")
train_df.head()
test_df.head()
train_labels_df.head()
pd.set_option('max_colwidth',150)

specs_df.head()
sample_submission_df.head()
print(f"train installation id: {train_df.installation_id.nunique()}")

print(f"test installation id: {test_df.installation_id.nunique()}")

print(f"test & submission installation ids identical: {set(test_df.installation_id.unique()) == set(sample_submission_df.installation_id.unique())}")
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(train_df)