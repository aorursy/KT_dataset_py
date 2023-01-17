# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import combinations



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/bakerybasket/bakeryBasket.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/bakerybasket/bakeryBasket.csv') 
dataset.head(5)
dataset['I1'],dataset['I2'],dataset['I3'],dataset['I4'] = np.nan,np.nan,np.nan,np.nan

for r in range(dataset.shape[0]):

    l = dataset.iloc[r,0].split(',')

    n = len(l)

    for i in range(1,n+1):

        dataset.iloc[r,i] = l[i-1]
dataset
#intializing minimum support

min_sup,records = 1,[]

for i in range(0,dataset.shape[0]):

    records.append([str(dataset.values[i,j]) for j in range(1,len(dataset.columns)) if str(dataset.values[i,j]) != 'nan'])

itemlist = sorted([item for sublist in records for item in sublist if item != np.nan])
#Frequent itemset when(k=1)

def freq_itm1(itemlist,min_sup):

    c1 = {i: itemlist.count(i) for i in itemlist}

    k1= {}

    for key,val in c1.items():

        if val >= min_sup:

            k1[key] = val

    return c1,k1



c1,k1= freq_itm1(itemlist,min_sup)

freq_item = pd.DataFrame(k1,index=['sup_count']).T



freq_item.sort_values(by=['sup_count'],inplace=True,ascending=False)



freq_item
#frequent itemset rule of pairing (k=2)

def check_freq(current,previous,n):

    if n > 1:

        subsets = list(combinations(current,n))

    else:

        subsets = current

    for item in subsets:

        if not item in previous:

            return False

        else:

            return True
def sub_list(item1,item2):

    return set(item1) <= set(item2)



def freq_itm2(k1,records,min_sup):

    k1 = sorted(list(k1.keys()))

    L1 = list(combinations(k1,2))

    c2,k2 = {},{}

    for it1 in L1:

        count = 0

        for it2 in records:

            if sub_list(it1,it2):

                count += 1

        c2[it1] = count

    for key,val in c2.items():

        if val >= min_sup:

            if check_freq(key,k1,1):

                k2[key] = val

    return c2,k2
c2,k2 = freq_itm2(k1,records,min_sup)

k2 = {key: value for key,value in k2.items() if value != 0}

freq_item2 = pd.DataFrame(k2,index=['sup_count']).T

freq_item2.sort_values(by=['sup_count'],inplace=True,ascending=False)

freq_item2
def freq_itm3(k2,records,min_sup):

    k2 = list(k2.keys())

    L2 = sorted(list(set([item for temp in k2 for item in temp])))

    L2 = list(combinations(L2,3))

    c3,k3 = {},{}

    for it1 in L2:

        count = 0

        for it2 in records:

            if sub_list(it1,it2):

                count += 1

        c3[it1] = count

    for key,val in c3.items():

        if val >= min_sup:

            if check_freq(key,k2,2):

                k3[key] = val

    return c3,k3

c3,k3 = freq_itm3(k2,records,min_sup)

k3 = {key: value for key,value in k3.items() if value != 0}

freq_item3= pd.DataFrame(k3,index=['sup_count']).T

freq_item3.sort_values(by=['sup_count'],inplace=True,ascending=False)

freq_item3
def freq_itm4(k3,records,min_sup):

    k3 = list(k3.keys())

    L3 = sorted(list(set([item for temp in k3 for item in temp])))

    L3 = list(combinations(L3,4))

    c4,k4 = {},{}

    for it1 in L3:

        count = 0

        for it2 in records:

            if sub_list(it1,it2):

                count += 1

        c4[it1] = count

        for key,val in c4.items():

            if val >= min_sup:

                if check_freq(key,k3,3):

                    k4[key] = val

    return c4,k4



# Test run

c4,k4 = freq_itm4(k3,records,min_sup)

k4 = {key: value for key,value in k4.items() if value != 0}

freq_item4 = pd.DataFrame(k4,index=['sup_count']).T

freq_item4
items = {**k1,**k2,**k3,**k4}
assc_sets = []

for it1 in list(k3.keys()):

    assc_subset = list(combinations(it1,2))

    assc_sets.append(assc_subset)



min_conf = 60

def sup_calc(it,items):

    return items[it]

# Calculating confidence

k3_assc = list(k3.keys())

selected_assc = []

for i in range(len(k3_assc)):

    for it1 in assc_sets[i]:

        denom = it1

        d = list(denom)

        num = set(k3_assc[i]) - set(it1)

        n = list(num)

        confidence = ((sup_calc(k3_assc[i],items))/(sup_calc(it1,items)))*100

        if confidence > min_conf:

            print("People who purchase {} and {} also purchase: {}".format(d[0],d[1],n[0]),"\n Confidence= {:.2f}%".format(confidence),"\n")
