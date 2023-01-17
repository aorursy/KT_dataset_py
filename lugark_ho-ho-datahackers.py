# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
n_children = 1000000 # n children to give

n_gift_type = 1000 # n types of gifts available

n_gift_quantity = 1000 # each type of gifts are limited to this quantity

n_gift_pref = 10 # number of gifts a child ranks

n_child_pref = 1000 # number of children a gift ranks

twins = int(0.004 * n_children)    # 0.4% of all population, rounded to the closest even number

ratio_gift_happiness = 2

ratio_child_happiness = 2



def avg_normalized_happiness(pred, child_pref, gift_pref):

    

    # check if number of each gift exceeds n_gift_quantity

    gift_counts = Counter(elem[1] for elem in pred)

    for count in gift_counts.values():

        assert count <= n_gift_quantity

                

    # check if twins have the same gift

    for t1 in range(0,twins,2):

        twin1 = pred[t1]

        twin2 = pred[t1+1]

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

        child_happiness = (n_gift_pref - np.where(gift_pref[child_id]==gift_id)[0]) * ratio_child_happiness

        if not child_happiness:

            child_happiness = -1



        gift_happiness = ( n_child_pref - np.where(child_pref[gift_id]==child_id)[0]) * ratio_gift_happiness

        if not gift_happiness:

            gift_happiness = -1



        total_child_happiness += child_happiness

        total_gift_happiness[gift_id] += gift_happiness

    

    # print(max_child_happiness, max_gift_happiness

    print('normalized child happiness=',float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) , \

        ', normalized gift happiness',np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity))

    return float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) + np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity)
def kid_happiness(kid_id, gift_id, wish):

    a = 20 - 2 * np.where(wish[kid_id] == gift_id)[0]

    if not a:

        return -1

    return a[0]



def gift_happiness(kid_id, gift_id, gift):

    a = 2000 - 2 * np.where(gift[gift_id] == kid_id)[0]

    if not a:

        return -1

    return a[0]
INPUT_PATH = '../input/'

           

wish = pd.read_csv(INPUT_PATH + 'child_wishlist.csv', header=None).as_matrix()[:, 1:]

gift = pd.read_csv(INPUT_PATH + 'gift_goodkids.csv', header=None).as_matrix()[:, 1:]

answ = np.zeros((len(wish)), dtype=np.int32)

answ[:] = -1

gift_count = np.zeros((len(gift)), dtype=np.int32)



print('twins')

for i in range(0, 4000, 2):

    gifts = set(wish[i, :])

    gifts.update(set(wish[i+1, :]))

    gifts = list(gifts)

    g = gifts[0]

    score = kid_happiness(i, g, wish) + kid_happiness(i+1, g, wish)

    for gg in gifts:

        c = kid_happiness(i, gg, wish) + kid_happiness(i+1, gg, wish)

        if c > score and gift_count[gg] < 1000:

            score = c

            g = gg

    answ[i] = g

    answ[i+1] = g

    gift_count[g] += 2



print('other children')

for k in range(10):

    for i in range(1000):

        for j in range(100):

            c = gift[i, k*100+j]

            if gift_count[i] < 1000 and answ[c] == -1:

                answ[c] = i

                gift_count[i] += 1

    for i in range(4000, len(answ)):

        g = wish[i, k]

        if gift_count[g] < 1000 and answ[i] == -1:

            answ[i] = g

            gift_count[g] += 1



print('unhappy children')

for i in range(4000, len(answ)):

    if answ[i] == -1:

        g = np.argmin(gift_count)

        answ[i] = g

        gift_count[g] += 1

        
ans1 = np.hstack((np.array(range(len(wish))).reshape((-1, 1)), answ.reshape((-1, 1))))

avg_normalized_happiness(ans1, gift, wish)
ans1
pd.DataFrame({'ChildId':ans1[:, 0], 'GiftId':ans1[:,1]}).to_csv(r'./subm', index=None)