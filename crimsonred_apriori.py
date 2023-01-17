# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib.pyplot as plt 

from scipy.special import comb

from itertools import combinations, permutations

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/groceries/groceries - groceries.csv')
items = (df['Item 1'].unique())
encoded_vals = []

for index, row in df.iterrows():

    labels = {}

    uncommons = list(set(items) - set(row))

    commons = list(set(items).intersection(row))

    for uc in uncommons:

        labels[uc] = 0

    for com in commons:

        labels[com] = 1

    encoded_vals.append(labels)

encoded_vals[0]

ohe_df = pd.DataFrame(encoded_vals)
freq_items = apriori(ohe_df, min_support=0.03, use_colnames=True)

freq_items.head(7)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.2)

rules.head(5)
rules.sort_values(by=['confidence'], ascending=False).head(5)