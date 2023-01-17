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
import missingno as msno

import pandas_profiling
# Read Training Data



train_df = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")

test_df = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")

sample_submission = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")

train_df.head(1)
train_df.tail(1)
train_df.columns
train_df.profile_report()
train_df[train_df['category']=="TECHNOLOGY"].iloc[1].to_dict()