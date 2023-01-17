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
!pip install apyori
from apyori import apriori
def load_dataset():

    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
x = load_dataset()
x
association_rules = list(apriori(x))
len(association_rules)
association_rules
for rule in association_rules:

    name_pair = list(rule[0])

    print("Rule:", name_pair)

    print("support",rule[1])
data = load_dataset()

data
def create_C1(dataset):

    C1=[]

    for transaction in dataset:

        for item in transaction:

            if not [item] in C1:

                C1.append([item])

    C1.sort()

    return list(map(frozenset, C1))
min_support = 2
def generate_L(D, CK, min_support):

    modified_dict={}

    for transaction in D:

        for grp in CK:

            if grp.issubset(transaction):

                if not grp in modified_dict:

                    modified_dict[grp] = 1

                else:

                    modified_dict[grp] += 1

    #print("CK with support:",modified_dict,"\n")

    r_list = []

    dict_with_freq = {}

    for key in modified_dict:

        supp = modified_dict[key]

        if supp>= min_support:

            r_list.insert(0,key)

            dict_with_freq[key] = supp

    

    return r_list,dict_with_freq

                    

            
D = list(map(set,data))
def create_CK(Lk, k):

    

    retList = []

    lenLk = len(Lk)

    for i in range(lenLk):

        for j in range(i+1, lenLk): 

            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]

            L1.sort(); L2.sort()

            if L1==L2:

                retList.append(Lk[i] | Lk[j])

    return retList
C1 = create_C1(data)

C1
L1, supp1 = generate_L(D,C1,min_support)

print(supp1)
C2 = create_CK(L1,2)

C2
L2, supp2 = generate_L(D,C2,min_support)

print(supp2)
C3 = create_CK(L2,3)

C3
L3, supp3 = generate_L(D,C3,min_support)

print(supp3)
C4 = create_CK(L3,4)

C4
C1 = create_C1(data)

CK = C1

i = 2

while CK !=[]:

    LK,supp = generate_L(D,CK,min_support)

    CK = create_CK(LK,i)

    i+=1

print(LK,"\n",supp)