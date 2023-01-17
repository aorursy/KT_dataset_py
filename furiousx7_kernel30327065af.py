# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
import csv
import os
from dateutil.parser import parse
import re
import datetime

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#print(os.listdir("../input"))
def strip_date(s):
    match = re.search('(\d{1,2}/\d{1,2}/\d{4})([a-z ]+)', s)
    date = datetime.datetime.strptime(match.groups()[0], '%d/%m/%Y').date()
    return date
def strip_text(s):
    match = re.search('(\d{1,2}/\d{1,2}/\d{4})([a-z ]+)', s)
    return match.groups()[1]

raw_data = pd.read_csv('../input/dataset.csv', header = None, sep='^')
col_count = max([row.count(',') for row in raw_data[0]])
raw_data = pd.read_csv('../input/dataset.csv', header = None, names = list(range(0,col_count)))
s1 = pd.Series(raw_data[0].apply(strip_date), dtype='datetime64')
s2 = pd.Series(raw_data[0].apply(strip_text))
cart_data = pd.concat([s1, s2, raw_data.iloc[:,1:]], axis=1, ignore_index=True)
cart_data.sample(10)
#Example from https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
dataset = [cart_data.loc[i].dropna()[1:-1].values for i in range(len(cart_data))]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

frequent_itemsets
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
np.array([[1,2,3],[4,5,6]]).transpose()
