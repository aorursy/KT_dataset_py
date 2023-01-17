# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

train_df.head()
# Quantitative (categorical) or qualitative

quantitative = [f for f in train_df.columns if train_df.dtypes[f] != 'object']

qualitative = [f for f in train_df.columns if train_df.dtypes[f] == 'object']
# Example of plotting the count of a particular feature

%matplotlib inline

train_df['MSZoning'].value_counts().plot('bar')
# What are these miscellaneous features?

set(train_df['MiscFeature'])

train_df[['MiscFeature','MiscVal']].dropna().head()
# Get column names where more than two unique values for categorical data

potential_dummy = [c for c in qualitative if len(train_df[c].dropna().unique()) > 2]

print(potential_dummy)