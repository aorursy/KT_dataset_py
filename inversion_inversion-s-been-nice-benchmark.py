# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
child_prefs = pd.read_csv('../input/child_wishlist.csv', header=None)

child_prefs = child_prefs.drop(0, axis=1).values



gift_prefs = pd.read_csv('../input/gift_goodkids.csv', header=None)

gift_prefs = gift_prefs.drop(0, axis=1).values
twins = 4000

n_children = child_prefs.shape[0]

n_gift_type = gift_prefs.shape[0]

n_gift_quantity = gift_prefs.shape[1]
def pick_first_choice(child_pref, avail_gifts):

    

    # preference list (of remaining available gifts)

    overlap = set(child_pref) & set(avail_gifts)

    child_pref_available = [x for x in child_pref if x in overlap] # preserves pref order

    

    try: # first pick on the list

        return child_pref_available[0]

    except: # if prefered gifts aren't available, pick first available

        return avail_gifts[0]
gift_matches = []

gift_counter = np.zeros(n_gift_type)



for child in range(n_children):

    

    if child < twins and child % 2 == 0: # twin 1

        avail_gifts = np.where(gift_counter < n_gift_quantity-1)[0]

        chosen_gift = pick_first_choice(child_prefs[child], avail_gifts)

        

    elif child < twins and child % 2 == 1: # twin 2

        chosen_gift = chosen_gift # pick same as twin 1

        

    else: # not twins

        avail_gifts = np.where(gift_counter < n_gift_quantity)[0]

        chosen_gift = pick_first_choice(child_prefs[child], avail_gifts)        



    gift_counter[chosen_gift] += 1

    gift_matches.append((child, chosen_gift))
p = pd.DataFrame(gift_matches, columns=['ChildId', 'GiftId']).set_index('ChildId')

p.to_csv('nice_inversion_benchmark.csv')