# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import combinations



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/supermarket/GroceryStoreDataSet.csv',sep=',',header=None,index_col=False)

data['I1'],data['I2'],data['I3'],data['I4'] = np.nan,np.nan,np.nan,np.nan

for r in range(data.shape[0]):

    l = data.iloc[r,0].split(',')

    n = len(l)

    for i in range(1,n+1):

        data.iloc[r,i] = l[i-1]

data
min_sup,records = 2,[]

for i in range(0,data.shape[0]):

    records.append([str(data.values[i,j]) for j in range(1,len(data.columns)) if str(data.values[i,j]) != 'nan'])

itemlist = sorted([item for sublist in records for item in sublist if item != np.nan])

records
def stage_1(itemlist,min_sup):

    c1 = {i: itemlist.count(i) for i in itemlist}

    l1 = {}

    for key,val in c1.items():

        if val >= min_sup:

            l1[key] = val

    return c1,l1



# Test run

c1,l1 = stage_1(itemlist,min_sup)

print(c1)

print(l1)



df_stage1 = pd.DataFrame(l1,index=['sup_count']).T

df_stage1
'''Function to check if for each subset of the current itemlist(k), whether the combination of k-1 items(previous grouping/pairing),

  belongs to the previous itemlist, so that it qualifies to be a frequent itemlist. 

  Arguments : current itemlist, previous itemlist, n(= k-1)'''

def check_freq(curr,prev,n):

    if n > 1:

        subsets = list(combinations(curr,n))

    else:

        subsets = curr

    for item in subsets:

        if not item in prev:

            return False

        else:

            return True



'''Function to check if i1 is a sublist/subset of i2'''

def sublist(i1,i2):

    return set(i1) <= set(i2)



def stage_2(l1,records,min_sup):

    l1 = sorted(list(l1.keys()))

    L1 = list(combinations(l1,2))

    c2,l2 = {},{}

    for it1 in L1:

        count = 0

        for it2 in records:

            if sublist(it1,it2):

                count += 1

        c2[it1] = count

    for key,val in c2.items():

        if val >= min_sup:

            if check_freq(key,l1,1):

                l2[key] = val

    return c2,l2



# Test run

c2,l2 = stage_2(l1,records,min_sup)

l2 = {key: value for key,value in l2.items() if value != 0}

print(c2)

print("\n",l2)

print("\nNo. of itemsets = {}, No. of frequent itemsets = {}".format(len(list(c2)),len(list(l2))))

df_stage2 = pd.DataFrame(l2,index=['sup_count']).T

df_stage2
def stage_3(l2,records,min_sup):

    l2 = list(l2.keys())

    L2 = sorted(list(set([item for temp in l2 for item in temp])))

    L2 = list(combinations(L2,3))

    c3,l3 = {},{}

    for it1 in L2:

        count = 0

        for it2 in records:

            if sublist(it1,it2):

                count += 1

        c3[it1] = count

    for key,val in c3.items():

        if val >= min_sup:

            if check_freq(key,l2,2):

                l3[key] = val

    return c3,l3



# Test run

c3,l3 = stage_3(l2,records,min_sup)

l3 = {key: value for key,value in l3.items() if value != 0}

print("CURRENT ITEMSETS : \n\n",c3)

print("\nCURRENT FREQUENT ITEMSETS : \n\n",l3)

print("\nNo. of itemsets = {}, No. of frequent itemsets = {}".format(len(list(c3)),len(list(l3))))

df_stage3 = pd.DataFrame(l3,index=['sup_count']).T

df_stage3
def stage_4(l3,records,min_sup):

    l3 = list(l3.keys())

    L3 = sorted(list(set([item for temp in l3 for item in temp])))

    L3 = list(combinations(L3,4))

    c4,l4 = {},{}

    for it1 in L3:

        count = 0

        for it2 in records:

            if sublist(it1,it2):

                count += 1

        c4[it1] = count

        for key,val in c4.items():

            if val >= min_sup:

                if check_freq(key,l3,3):

                    l4[key] = val

    return c4,l4



# Test run

c4,l4 = stage_4(l3,records,min_sup)

l4 = {key: value for key,value in l4.items() if value != 0}

print("CURRENT ITEMSETS : \n\n",c4)

print("\nCURRENT FREQUENT ITEMSETS : \n\n",l4)

print("\nNo. of itemsets = {}, No. of frequent itemsets = {}".format(len(list(c4)),len(list(l4))))

df_stage4 = pd.DataFrame(l4,index=['sup_count']).T

df_stage4
items = {**l1,**l2,**l3,**l4}

items
'''Working on l3 to break the triplets to form dual pair + individual item comnbination sets for forming the association rules (like, {A,B,C} => {A,B} --> {C} and more)'''

assc_sets = []

for it1 in list(l3.keys()):

    assc_subset = list(combinations(it1,2))

    assc_sets.append(assc_subset)



'''Implementing the association rule.

   An association rule is formed iff the confidence of that rule exceeds the minimum confidence threshold.

   Assuming minimum confidence = 50%

'''

min_conf = 50

# Function to calculate support score

def sup_calc(it,items):

    return items[it]

# Calculating confidence

l3_assc = list(l3.keys())

selected_assc = []

for i in range(len(l3_assc)):

    for it1 in assc_sets[i]:

        denom = it1

        d = list(denom)

        num = set(l3_assc[i]) - set(it1)

        n = list(num)

        confidence = ((sup_calc(l3_assc[i],items))/(sup_calc(it1,items)))*100

        if confidence > min_conf:

            print("Confidence of the association rule {} --> {} = {:.2f}%".format(denom,num,confidence))

            print("STATUS : SELECTED RULE\n* People who buy {} and {} also tend to buy : {} *\n".format(d[0],d[1],n[0]))

        else:

            print("Confidence of the association rule {} --> {} = {:.2f}%".format(denom,num,confidence))

            print("STATUS : REJECTED RULE\n")