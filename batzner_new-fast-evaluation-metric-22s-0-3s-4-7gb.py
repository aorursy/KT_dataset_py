import numpy as np

import pandas as pd

import math

from tqdm import tqdm



n_children = 1000000  # n children to give

n_gift_type = 1000  # n types of gifts available

n_gift_quantity = 1000  # each type of gifts are limited to this quantity

n_gift_pref = 100  # number of gifts a child ranks

n_child_pref = 1000  # number of children a gift ranks

twins = math.ceil(0.04 * n_children / 2.) * 2  # 4% of all population, rounded to the closest number

triplets = math.ceil(0.005 * n_children / 3.) * 3  # 0.5% of all population, rounded to the closest number

ratio_gift_happiness = 2

ratio_child_happiness = 2



child_wishlists = pd.read_csv('../input/child_wishlist_v2.csv', header=None).drop(0, 1).values

gift_goodkids = pd.read_csv('../input/gift_goodkids_v2.csv', header=None).drop(0, 1).values

random_sub = pd.read_csv('../input/sample_submission_random_v2.csv').values.tolist()



max_child_happiness = n_gift_pref * ratio_child_happiness

max_gift_happiness = n_child_pref * ratio_gift_happiness





def lcm(a, b):

    """Compute the lowest common multiple of a and b"""

    # in case of large numbers, using floor division

    return a * b // math.gcd(a, b)





# to avoid float rounding error

# find common denominator

# NOTE: I used this code to experiment different parameters, so it was necessary to get the multiplier

# Note: You should hard-code the multipler to speed up, now that the parameters are finalized

denominator1 = n_children * max_child_happiness

denominator2 = n_gift_quantity * max_gift_happiness * n_gift_type

common_denom = lcm(denominator1, denominator2)

multiplier = common_denom / denominator1
%%time

# Construct the score matrices

child_happiness = np.full((n_gift_type, n_children), -1 * multiplier, dtype=np.int16)

gift_happiness = np.full((n_gift_type, n_children), -1, dtype=np.int16)



# Iterate the wishlists

to_add = (np.arange(n_gift_pref, 0, -1) * ratio_child_happiness + 1) * int(multiplier)

for child, wishlist in tqdm(enumerate(child_wishlists)):

    child_happiness[wishlist, child] += to_add



# Iterate the goodkids

to_add = np.arange(n_child_pref, 0, -1) * ratio_gift_happiness + 1

for gift, goodkids in tqdm(enumerate(gift_goodkids)):

    gift_happiness[gift, goodkids] += to_add
def avg_normalized_happiness(children, gifts):

    total_child_happiness = np.sum(child_happiness[gifts, children])

    total_gift_happiness = np.sum(gift_happiness[gifts, children])

        

    return float(math.pow(total_child_happiness, 3) 

                 + math.pow(total_gift_happiness, 3)) / math.pow(common_denom, 3)
%%time

children, gifts = zip(*random_sub)

for _ in range(100):

    score = avg_normalized_happiness(children, gifts)

print('ANH', score)
from collections import Counter



def avg_normalized_happiness(pred):

    # check if number of each gift exceeds n_gift_quantity

    gift_counts = Counter(elem[1] for elem in pred)

    for count in gift_counts.values():

        assert count <= n_gift_quantity



    # check if triplets have the same gift

    for t1 in np.arange(0, triplets, 3):

        triplet1 = pred[t1]

        triplet2 = pred[t1 + 1]

        triplet3 = pred[t1 + 2]

        # print(t1, triplet1, triplet2, triplet3)

        assert triplet1[1] == triplet2[1] and triplet2[1] == triplet3[1]



    # check if twins have the same gift

    for t1 in np.arange(triplets, triplets + twins, 2):

        twin1 = pred[t1]

        twin2 = pred[t1 + 1]

        # print(t1)

        assert twin1[1] == twin2[1]



    max_child_happiness = n_gift_pref * ratio_child_happiness

    max_gift_happiness = n_child_pref * ratio_gift_happiness

    total_child_happiness = 0

    total_gift_happiness = np.zeros(n_gift_type)



    for row in pred:

        child_id = row[0]

        gift_id = row[1]



        # check if child_id and gift_id exist

        assert child_id < n_children

        assert gift_id < n_gift_type

        assert child_id >= 0

        assert gift_id >= 0

        child_happiness = (n_gift_pref - np.where(child_wishlists[child_id] == gift_id)[0]) * ratio_child_happiness

        if not child_happiness:

            child_happiness = -1



        gift_happiness = (n_child_pref - np.where(gift_goodkids[gift_id] == child_id)[0]) * ratio_gift_happiness

        if not gift_happiness:

            gift_happiness = -1



        total_child_happiness += child_happiness

        total_gift_happiness[gift_id] += gift_happiness



    # to avoid float rounding error

    # find common denominator

    # NOTE: I used this code to experiment different parameters, so it was necessary to get the multiplier

    # Note: You should hard-code the multipler to speed up, now that the parameters are finalized

    denominator1 = n_children * max_child_happiness

    denominator2 = n_gift_quantity * max_gift_happiness * n_gift_type

    common_denom = lcm(denominator1, denominator2)

    multiplier = common_denom / denominator1



    # # usually denom1 > demon2

    return float(math.pow(total_child_happiness * multiplier, 3) + math.pow(np.sum(total_gift_happiness), 3)) / float(

        math.pow(common_denom, 3))

    # return math.pow(float(total_child_happiness)/(float(n_children)*float(max_child_happiness)),2) + math.pow(np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity),2)
%%time

for _ in range(10):

    score = avg_normalized_happiness(random_sub)

print('ANH', score)