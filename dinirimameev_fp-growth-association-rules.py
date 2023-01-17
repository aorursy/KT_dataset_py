# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_excel('../input/online-retail-dataset/online_retail_II.xlsx')

dataset.head()
# Group products by transaction

grouped_prods = dataset.groupby('Invoice')['Description'].apply(lambda group_series: group_series.tolist()).reset_index()

groups_lists = grouped_prods['Description'].values.tolist()

# Convert table to list

# Set threshold of count to 2

data = list(filter(lambda x: len(x) > 2, groups_lists))
from mlxtend.preprocessing import TransactionEncoder



te = TransactionEncoder()

te_ary = te.fit(data).transform(data)

df = pd.DataFrame(te_ary, columns=te.columns_)

df
# use fp-growth algorithm

from mlxtend.frequent_patterns import fpgrowth



f_patterns = fpgrowth(df, min_support=0.005, use_colnames=True)

f_patterns
import itertools as it

from itertools import *



# help function

def partition(pred, iterable):

    t1, t2 = it.tee(iterable)

    return it.filterfalse(pred, t1), filter(pred, t2)



# divides list on all possible pairs

def part2(el_list):

    pairs = [[[x[1] for x in f] for f in partition(lambda x: x[0], zip(pattern, el_list))] \

     for pattern in product([True, False], repeat=len(el_list))]

    # remove pairs as [] -> [some content], [some content] -> []

    return pairs[1:-1]
# convert dataframe to dictionary

supports = f_patterns['support'].to_list()

itemsets = f_patterns['itemsets'].to_list()



patterns_dict = {}

for x in range(len(itemsets)):

    patterns_dict[tuple(sorted(itemsets[x]))] = supports[x]

    

# generate asssociation_rules

as_rules_dict = {'left': [], 'right': [], 'confidence': []}

for pattern, support in patterns_dict.items():

    if len(pattern) > 1:

        upper_support = support

        as_rules = part2(pattern)

        

        for as_r in as_rules:

            left_part = sorted(as_r[0])

            right_part = as_r[1]

            lower_support = patterns_dict[tuple(left_part)]

            conf = upper_support / lower_support

            

            as_rules_dict['left'].append(left_part)

            as_rules_dict['right'].append(right_part)

            as_rules_dict['confidence'].append(conf)

            



strong_as_rules = pd.DataFrame.from_dict(as_rules_dict)

# sort by confidence, remove all rules with confidence lower than 0.8

strong_as_rules = strong_as_rules.sort_values('confidence', ascending=False)

strong_as_rules = strong_as_rules[strong_as_rules['confidence'] > 0.8]



strong_as_rules
import io



# Save results

output = io.BytesIO()



# Use the StringIO object as the filehandle.

writer = pd.ExcelWriter(output, engine='xlsxwriter')



strong_as_rules.to_excel('association_rules_online_retail_II.xlsx')

writer.save()