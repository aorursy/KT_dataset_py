# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

groceries_file = pd.read_csv('/kaggle/input/groceries/groceries.csv',names=["Items"],header = None , sep =";")

groceries_file.head()
data = []

sentences = list(groceries_file['Items'])

for sen in sentences:

    data.append(sen)

    

data = '\n'.join(data)
from collections import defaultdict



D2 = defaultdict (int) # Empty dictionary



D2['existing-key'] = 5 # Create one key-value pair



D2['existing-key'] += 1 # Update

D2['new-key'] += 1



print (D2)
from collections import defaultdict

from itertools import combinations, permutations # Hint!



def update_pair_counts (pair_counts, itemset):

    """

    Updates a dictionary of pair counts for

    all pairs of items in a given itemset.

    """

    assert type (pair_counts) is defaultdict



    #for a in list(permutations(itemset, 2)):

        #pair_counts[a] += 1

    for (a,b) in combinations(itemset, 2):

        pair_counts[(a,b)] += 1

        pair_counts[(b,a)] += 1
def update_item_counts(item_counts, itemset):

    for i in itemset:

        item_counts[i] += 1
def filter_rules_by_conf (pair_counts, item_counts, threshold):

    rules = {} # (item_a, item_b) -> conf (item_a => item_b)

    for (a,b) in pair_counts:

        conf = pair_counts[(a,b)]/item_counts[a]

        if conf>=threshold:

            rules[(a,b)] = conf

    return rules
def gen_rule_str(a, b, val=None, val_fmt='{:.3f}', sep=" = "):

    text = "{} => {}".format(a, b)

    if val:

        text = "conf(" + text + ")"

        text += sep + val_fmt.format(val)

    return text



def print_rules(rules):

    if type(rules) is dict or type(rules) is defaultdict:

        from operator import itemgetter

        ordered_rules = sorted(rules.items(), key=itemgetter(1), reverse=True)

    else: # Assume rules is iterable

        ordered_rules = [((a, b), None) for a, b in rules]

    for (a, b), conf_ab in ordered_rules:

        print(gen_rule_str(a, b, conf_ab))
def find_assoc_rules(receipts, threshold):

    pc = defaultdict(int)

    ic = defaultdict(int)

    for itemset in receipts:

        update_pair_counts(pc,itemset)

        update_item_counts(ic,itemset)

    rules = filter_rules_by_conf(pc,ic,threshold)

    return rules
def intersect_keys(d1, d2):

    assert type(d1) is dict or type(d1) is defaultdict

    assert type(d2) is dict or type(d2) is defaultdict

    return set(d1.keys()) & set(d2.keys())
# Confidence threshold

THRESHOLD = 0.5



# Only consider rules for items appearing at least `MIN_COUNT` times.

MIN_COUNT = 10
#split file by line

splittedList = data.splitlines()





#list of receipts

commaseplist = []

for a in splittedList:

    commaseplist.append(set(a.split(',')))



#update counts

pc = defaultdict(int)

ic = defaultdict(int)

for itemset in commaseplist:

    update_pair_counts(pc,itemset)

    update_item_counts(ic,itemset)



#rules

basket_rules = {}

for (a,b) in pc:

    conf = pc[(a,b)]/ic[a]

    if conf>=THRESHOLD and ic[a]>=MIN_COUNT:

        basket_rules[(a,b)] = conf

print(basket_rules)
### `basket_rules_test`: TEST CODE ###

print("Found {} rules whose confidence exceeds {}.".format(len(basket_rules), THRESHOLD))

print("Here they are:\n")

print_rules(basket_rules)



assert len(basket_rules) == 19

assert all([THRESHOLD <= v < 1.0 for v in basket_rules.values()])

ans_keys = [("pudding powder", "whole milk"), ("tidbits", "rolls/buns"), ("cocoa drinks", "whole milk"), ("cream", "sausage"), ("rubbing alcohol", "whole milk"), ("honey", "whole milk"), ("frozen fruits", "other vegetables"), ("cream", "other vegetables"), ("ready soups", "rolls/buns"), ("cooking chocolate", "whole milk"), ("cereals", "whole milk"), ("rice", "whole milk"), ("specialty cheese", "other vegetables"), ("baking powder", "whole milk"), ("rubbing alcohol", "butter"), ("rubbing alcohol", "citrus fruit"), ("jam", "whole milk"), ("frozen fruits", "whipped/sour cream"), ("rice", "other vegetables")]

for k in ans_keys:

    assert k in basket_rules



print("\n(Passed!)")