from collections import OrderedDict

from pathlib import Path

import operator as op

from functools import reduce



import pandas as pd

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns
ROOT = Path('/kaggle/input/random-shopping-cart/')



MIN_SUPPORT = 0.1
def generate_c1(basket):

    c1 = OrderedDict()

    num_transactions = len(basket)

    

    # Find and count all the items in the basket

    for transaction in basket:

        for item in transaction:

            if tuple([item]) in c1.keys():

                c1[tuple([item])] += 1 / num_transactions

            else:

                c1[tuple([item])] = 0



    return c1



def ncr(n, r):

    r = min(r, n-r)

    numer = reduce(op.mul, range(n, n-r, -1), 1)

    denom = reduce(op.mul, range(1, r+1), 1)

    return numer // denom



def is_acceptable(list_of_lk, list_of_items):

    len_list_of_items = len(list_of_items)

    num_comb = ncr(len_list_of_items, len_list_of_items-1)

    if len(list_of_lk) < num_comb:

        return False



    set_of_items = set(list_of_items)



    i = 0

    for candidate in list_of_lk:

        if set(candidate).issubset(set_of_items):

            i += 1



    if i == num_comb:

        return True

    else:

        return False



def generate_ck(basket, lk_minus_1):

    list_of_lk_minus_1 = list(lk_minus_1.keys())

    len_list_of_lk_minus_1 = len(list_of_lk_minus_1)

    num_transactions = len(basket)

    

    ck = OrderedDict()



    for i in range(len_list_of_lk_minus_1 - 1):

        for j in range(i + 1, len_list_of_lk_minus_1):

            first_list = sorted(list_of_lk_minus_1[i])

            second_list = sorted(list_of_lk_minus_1[j])

            

            if first_list[:-1] == second_list[:-1]:

                new_list = sorted(list(first_list) + [second_list[-1]])



                # Pruning

                if is_acceptable(list_of_lk_minus_1, new_list):

                    # Count the number of occurrences

                    k = 0

                    for transaction in basket:

                        if set(new_list).issubset(transaction):

                            k += 1 / num_transactions



                    ck[tuple(new_list)] = k



    return ck



def get_lk(ck_minus_1, min_sup):

    return {k: v for k, v in ck_minus_1.items() if v >= min_sup}
df = pd.read_csv(ROOT / 'dataset_group.csv', header=None, names=['date', 'transaction_id', 'item'])
df
basket = []



for transaction_id in tqdm(df.transaction_id.unique()):

    items = []

    for item in df[df.transaction_id == transaction_id].item:

        items.append(item)

        

    basket.append(items)
len_trans = [len(tran) for tran in basket]
print('min:', min(len_trans))

print('max:', max(len_trans))

sns.distplot(len_trans)

plt.show()
ci = generate_c1(basket)

li = get_lk(ci, MIN_SUPPORT)



all_l = li.copy()



while len(li) > 0:

    print(len(li))

    li_minus_1 = li.copy()

    ci = generate_ck(basket, li)

    li = get_lk(ci, MIN_SUPPORT)

    

    all_l.update(li)
# for list_of_items, support in li_minus_1.items():

#     print(f'{list_of_items} | {support}')
result_dict = {}



for idx, col in enumerate(np.array(list(li_minus_1.keys())).T):

    result_dict[f'item {idx+1}'] = col

    

result_dict['support'] = list(li_minus_1.values())
result_df = pd.DataFrame(result_dict).sort_values('support', ascending=False, ignore_index=True)
result_df
l1 = {k: v for k, v in all_l.items() if len(k) == 1}

l2 = {k: v for k, v in all_l.items() if len(k) == 2}
confidence_dict = dict()



for list_of_items, support in l2.items():

    for item in list_of_items:

        confidence_dict[tuple([item] + list(set(list_of_items) - set([item])))] = support / l1[(item,)]
# confidence_dict
confidence_df = pd.DataFrame({

    'rule': [' => '.join(pair) for pair in confidence_dict.keys()],

    'confidence': list(confidence_dict.values())

}).sort_values('confidence', ascending=False, ignore_index=True)
confidence_df