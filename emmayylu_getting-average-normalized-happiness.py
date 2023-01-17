# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



start_time = time.time()



#load data

child_table = pd.read_csv("../input/child_wishlist_v2.csv",header = None).drop(0,axis = 1)

gift_table = pd.read_csv("../input/gift_goodkids_v2.csv", header = None).drop(0,axis = 1)

samples_table = pd.read_csv("../input/sample_submission_random_v2.csv")

#write down all the condition

n_children = 1000000

n_gift_type = 1000 #number of types of gifts

n_gift_each = 1000 #number of gift in each type

n_gift_pref = 100 #number of gifts a child ranks

n_child_pref = 1000 #number of children a gift ranks

n_twins = 0.04 *n_children

n_triplets = 0.005 *n_children



children = [i for i in range(0,1000000)]

assigned = samples_table['GiftId']
#getting the sum of all children happiness

def chappiness(samples):

    sch = 0

    for child in children:

        assigned = samples.iloc[child]['GiftId'].item()

        child_pref = child_table.iloc[child,:].tolist()

        #if the gift is in in the preference list 

        if assigned in child_pref:

            fil = child_table.iloc[child] == assigned

            index_gift = child_table.loc[child,fil].index.item()-1

            sch = sch +(n_gift_pref - index_gift) *2

        else:

            sch += -1     

    return sch
#getting the sum of all gift happiness

def ghappiness(sample):

    sgh = 0

    #for each child:

    for child in children:

        received = samples_table.iloc[child,1]

        good_children = gift_table.iloc[received,:].tolist()

        #if he/she is in the good kid list for the gift:

        if child in good_children:

            fil = gift_table.iloc[received] == child

            child_order = gift_table.loc[received,fil].index.item()-1

            sgh = sgh + 2* (n_child_pref - child_order) 

        #if he/she is not in the good kid list for the gift:

        if child not in good_children:

            sgh = sgh -1

    return sgh

def result_anh(sample):

    sch = chappiness(sample)

    sgh = ghappiness(sample)

    sum_max_ch = n_children * n_gift_pref *2

    sum_max_gh = n_children * n_child_pref *2

    anch = sch/sum_max_ch

    ansh = sgh/sum_max_gh

    anh = anch**3 +ansh **3

    return anh
result_anh(samples_table)



end_time = time.time()

exec_time = end_time - start_time

print(exec_time)