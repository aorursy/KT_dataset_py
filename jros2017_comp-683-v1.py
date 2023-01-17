# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori, association_rules

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

dataset = '../input/survey_results_public.csv'

columns = ['Employment', 'FormalEducation', 'UndergradMajor', 'CareerSatisfaction', 'HopeFiveYears', 'YearsCodingProf', 'DevType']

df = pd.read_csv(dataset,
                 dtype='str',
                 na_values=['NA'],
                 usecols=columns)

df = pd.read_csv(dataset, usecols=columns, dtype=str)
df = pd.concat([df[columns], df['DevType'].str.split(';', expand=True)], axis=1)

df = pd.get_dummies(df)

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

#with open('itemsets', 'w') as outfile:
#    frequent_itemsets.to_string(outfile)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

#with open('rules', 'w') as outfile:
#    rules.to_string(outfile)

print(rules)

