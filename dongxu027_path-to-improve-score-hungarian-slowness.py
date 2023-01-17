import os, operator, math

import pandas as pd

import numpy as np

import datetime as dt

from tqdm import tqdm

import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from collections import defaultdict, Counter
child_data = pd.read_csv('../input/santa-gift-matching/child_wishlist_v2.csv', 

                         header=None).drop(0, 1).values

gift_data = pd.read_csv('../input/santa-gift-matching/gift_goodkids_v2.csv', 

                        header=None).drop(0, 1).values



n_children = 1000000

n_gift_type = 1000 

n_gift_quantity = 1000

n_child_wish = 100

triplets = 5001

twins = 40000

tts = triplets + twins 
gift_happiness = (1. / (2 * n_gift_type)) * np.ones(

    shape=(n_gift_type, n_children), dtype = np.float32)



for g in range(n_gift_type):

    for i, c in enumerate(gift_data[g]):

        gift_happiness[g,c] = -2. * (n_gift_type - i)  



child_happiness = (1. / (2 * n_child_wish)) * np.ones(

    shape=(n_children, n_gift_type), dtype = np.float32)



for c in range(n_children):

    for i, g in enumerate(child_data[c]):

        child_happiness[c,g] = -2. * (n_child_wish - i) 



gift_ids = np.array([[g] * n_gift_quantity for g in range(n_gift_type)]).flatten()
def lcm(a, b):

    """Compute the lowest common multiple of a and b"""

    # in case of large numbers, using floor division

    return a * b // math.gcd(a, b)



def avg_normalized_happiness(pred, gift, wish):

    

    n_children = 1000000 

    n_gift_type = 1000 

    n_gift_quantity = 1000 

    n_gift_pref = 100 

    n_child_pref = 1000

    twins = math.ceil(0.04 * n_children / 2.) * 2   

    triplets = math.ceil(0.005 * n_children / 3.) * 3   

    ratio_gift_happiness = 2

    ratio_child_happiness = 2



    # check if triplets have the same gift

    for t1 in np.arange(0, triplets, 3):

        triplet1 = pred[t1]

        triplet2 = pred[t1+1]

        triplet3 = pred[t1+2]

        assert triplet1 == triplet2 and triplet2 == triplet3

                

    # check if twins have the same gift

    for t1 in np.arange(triplets, triplets+twins, 2):

        twin1 = pred[t1]

        twin2 = pred[t1+1]

        assert twin1 == twin2



    max_child_happiness = n_gift_pref * ratio_child_happiness

    max_gift_happiness = n_child_pref * ratio_gift_happiness

    total_child_happiness = 0

    total_gift_happiness = np.zeros(n_gift_type)

    

    for i in range(len(pred)):

        child_id = i

        gift_id = pred[i]

        

        # check if child_id and gift_id exist

        assert child_id < n_children

        assert gift_id < n_gift_type

        assert child_id >= 0 

        assert gift_id >= 0

        child_happiness = (n_gift_pref - np.where(wish[child_id]==gift_id)[0]

                          ) * ratio_child_happiness

        if not child_happiness:

            child_happiness = -1



        gift_happiness = ( n_child_pref - np.where(gift[gift_id]==child_id)[0]

                         ) * ratio_gift_happiness

        if not gift_happiness:

            gift_happiness = -1



        total_child_happiness += child_happiness

        total_gift_happiness[gift_id] += gift_happiness

        

    denominator1 = n_children*max_child_happiness

    denominator2 = n_gift_quantity*max_gift_happiness*n_gift_type

    common_denom = lcm(denominator1, denominator2)

    multiplier = common_denom / denominator1



    ret = float(math.pow(total_child_happiness*multiplier,3) 

                + math.pow(np.sum(total_gift_happiness),3)

               ) / float(math.pow(common_denom,3))



    return ret
initial_sub = '../input/max-flow-with-min-cost-v2-0-9267/subm_0.926447635166.csv'

subm = pd.read_csv(initial_sub)

subm['gift_rank'] = subm.groupby('GiftId').rank() - 1

subm['gift_id'] = subm['GiftId'] * 1000 + subm['gift_rank']

subm['gift_id'] = subm['gift_id'].astype(np.int64)

current_gift_ids = subm['gift_id'].values
wish = pd.read_csv('../input/santa-gift-matching/child_wishlist_v2.csv', 

                   header=None).as_matrix()[:, 1:]

gift_init = pd.read_csv('../input/santa-gift-matching/gift_goodkids_v2.csv', 

                        header=None).as_matrix()[:, 1:]

gift = gift_init.copy()

answ_org = np.zeros(len(wish), dtype=np.int32)

answ_org[subm[['ChildId']]] = subm[['GiftId']]

score_org = avg_normalized_happiness(answ_org, gift, wish)

print('Predicted score: {:.8f}'.format(score_org))
def optimize_block(child_block, current_gift_ids):

    gift_block = current_gift_ids[child_block]

    C = np.zeros((block_size, block_size))

    for i in range(block_size):

        c = child_block[i]

        for j in range(block_size):

            g = gift_ids[gift_block[j]]

            C[i, j] = child_happiness[c][g]

    row_ind, col_ind = linear_sum_assignment(C)

    return (child_block[row_ind], gift_block[col_ind])
block_size = 1500

n_blocks = int((n_children - tts) / block_size)

children_rmd = 1000000 - 45001 - n_blocks * block_size

print('block size: {}, num blocks: {}, children reminder: {}'.

      format(block_size, n_blocks, children_rmd))
answ_iter = np.zeros(len(wish), dtype=np.int32)

score_best = score_org

subm_best = subm

perm_len = 2

block_len = 5

for i in range(perm_len):  

    print('Current permutation step is: %d' %(i+1))

    child_blocks = np.split(np.random.permutation

                            (range(tts, n_children - children_rmd)), n_blocks)

    for child_block in tqdm(child_blocks[:block_len]):

        start_time = dt.datetime.now()

        cids, gids = optimize_block(child_block, current_gift_ids=current_gift_ids)

        current_gift_ids[cids] = gids

        end_time = dt.datetime.now()

        print('Time spent to optimize this block in seconds: {:.2f}'.

              format((end_time-start_time).total_seconds()))

        ## need evaluation step for every block iteration 

        subm['GiftId'] = gift_ids[current_gift_ids]

        answ_iter[subm[['ChildId']]] = subm[['GiftId']]

        score_iter = avg_normalized_happiness(answ_iter, gift, wish)

        print('Score achieved in current iteration: {:.8f}'.format(score_iter))

        if score_iter > score_best:

            subm_best['GiftId'] = gift_ids[current_gift_ids]

            score_best = score_iter

            print('This is a performance improvement!')

        else: print('No improvement at this iteration!')

            

subm_best[['ChildId', 'GiftId']].to_csv('improved_sub.csv', index=False)

print('Best score achieved is: {:.8f}'.format(score_best))