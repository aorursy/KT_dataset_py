import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import itertools
data = pd.read_csv('/kaggle/input/data.csv')
rowlength = len(data)

no_features = len(data.values[0])
data.head()
print(rowlength, no_features)
minimum_support_count = 2



records = []

for i in range(0, rowlength):

    records.append([str(data.values[i,j]) for j in range(0, no_features)])

items = sorted([item for sublist in records for item in sublist if item != 'Nan'])
def stage_1(items, minimum_support_count):

    c1 = {i:items.count(i) for i in items}

    l1= {}

    

    for key, value in c1.items():

        if(value >= minimum_support_count):

            l1[key] = value

    return c1, l1

c1, l1 = stage_1(items, minimum_support_count)
c1
l1
def sublist(lst1,lst2):

    return(set(lst1) <= set(lst2))
def check_subset_frequency(itemset, l, n):

    if(n>1):

        subsets = list(itertools.combinations(itemset,n))

    else:

        subsets = itemset

    for iter1 in subsets:

        if not iter1 in l:

            return False

    return True
def stage_2(l1, records, minimum_support_count):

    l1 = sorted(list(l1.keys()))

    l1 = list(itertools.combinations(l1,2))

    c2 = {}

    l2 = {}

    for iter1 in l1:

        count = 0

        for iter2 in records:

            if(sublist(iter1,iter2)):

                count += 1

        c2[iter1] = count

    for key,value in c2.items():

        if(value >= minimum_support_count):

            if(check_subset_frequency(key, l1,2)):

                l2[key] = value

    return c2, l2



c2,l2 = stage_2(l1, records, minimum_support_count)
c2
l2