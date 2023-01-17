import numpy as np

import pandas as pd

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 13, 6

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df_child_pref = pd.read_csv('../input/santa-gift-matching/child_wishlist_v2.csv',header=None)

df_gift_pref = pd.read_csv('../input/santa-gift-matching/gift_goodkids_v2.csv',header=None)

df_result = pd.read_csv('../input/max-flow-with-min-cost-v2-0-9267/subm_0.926447635166.csv')
child_pref = df_child_pref.drop(0, 1).values

gift_pref = df_gift_pref.drop(0, 1).values

results = df_result.values.tolist()
df_child_pref.head()
df_gift_pref.head()
df_result.head()
n_children = 1000000 # n children to give

n_gift_type = 1000 # n types of gifts available

n_gift_quantity = 1000 # each type of gifts are limited to this quantity

n_gift_pref = 100 # number of gifts a child ranks

n_child_pref = 1000 # number of children a gift ranks

ratio_gift_happiness = 2

ratio_child_happiness = 2
max_child_happiness = n_gift_pref * ratio_child_happiness

max_gift_happiness = n_child_pref * ratio_gift_happiness

total_max_gift_happiness = max_gift_happiness * n_gift_type * n_gift_quantity

total_max_child_happiness = max_child_happiness * n_children

max_happiness_r = total_max_gift_happiness // total_max_child_happiness; max_happiness_r
# Child and gift happiness points

def happiness_points_all(pred, child_pref, gift_pref):

    child_happiness_list = np.zeros(len(pred), dtype=np.int)

    gift_happiness_list = np.zeros(len(pred), dtype=np.int)

    for i, row in enumerate(pred):

        child_id = row[0]

        gift_id = row[1]

        child_happiness = (n_gift_pref - np.where(gift_pref[child_id]==gift_id)[0]) * ratio_child_happiness

        if not child_happiness:

            child_happiness = -1

        gift_happiness = ( n_child_pref - np.where(child_pref[gift_id]==child_id)[0]) * ratio_gift_happiness

        if not gift_happiness:

            gift_happiness = -1

        child_happiness_list[i] = child_happiness

        gift_happiness_list[i] = gift_happiness        

    return child_happiness_list, gift_happiness_list
# Score function using happiness point lists

import math

def avg_normalized_happiness_from_list(child_happiness_list, gift_happiness_list):

    total_child_happiness = np.sum(child_happiness_list)

    total_gift_happiness = np.sum(gift_happiness_list)

    total_child_happiness

    return float(math.pow(total_child_happiness * max_happiness_r,3) + math.pow(np.sum(total_gift_happiness),3)) / float(math.pow(total_max_gift_happiness,3))
# Happiness points

child_happiness_list, gift_happiness_list = happiness_points_all(results, gift_pref, child_pref)
total_child_happiness = np.sum(child_happiness_list)

total_gift_happiness = np.sum(gift_happiness_list)

total_child_happiness, total_gift_happiness
# The Score

avg_normalized_happiness_from_list(child_happiness_list, gift_happiness_list)
# Effective ratio between child and gift happiness points

eff_r = math.pow(max_happiness_r, 3) * math.pow(total_child_happiness / total_gift_happiness, 2); 

eff_r
# Combined linearized happiness

child_gift_happiness_list = child_happiness_list * eff_r + gift_happiness_list; 

child_gift_happiness_list
# Number of completely unhappy children

np.sum(child_happiness_list < 0)
# Number of completely unhappy gifts

np.sum(gift_happiness_list < 0)
# Childern / Gift unhappy rate

np.sum(child_happiness_list < 0) / n_children, np.sum(gift_happiness_list < 0) / n_children
# Effective happiness unhappy rate

np.sum(child_gift_happiness_list < 0) / n_children
plt.hist(child_happiness_list, bins=22); plt.yscale('log'); 

plt.ylabel('N of children'); plt.xlabel('Child happiness points'); 

plt.title('Child happiness'); plt.show()
plt.hist(gift_happiness_list, bins=100); plt.yscale('log')

plt.ylabel('N of children'); plt.xlabel('Gift happiness points'); 

plt.title('Gift happiness'); plt.show()
plt.hist(child_gift_happiness_list, bins=100); plt.yscale('log')

plt.ylabel('N of children'); plt.xlabel('Combined Linearized happiness'); 

plt.title('Overall happiness'); plt.show()