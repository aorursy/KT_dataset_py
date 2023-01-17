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
# Load in Train Data

df_train = pd.read_csv('../input/train.csv')

df_train.head()
# Load in Test Data

df_test = pd.read_csv('../input/test.csv')

df_test.head()
df_train['NA_Sales'].mean()
df_ss = pd.read_csv('../input/df_sample_submission.csv')

df_ss
df_ss['Prediction'] = df_train['NA_Sales'].mean()

df_ss
# Save to csv because I want to submit my score

df_ss.to_csv('submission.csv', index=False)