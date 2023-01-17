import numpy as np

import pandas as pd
%%time

child_prefs = pd.read_csv('../input/santa-gift-matching/child_wishlist_v2.csv', header=None).drop(0, axis=1).values

gift_prefs = pd.read_csv('../input/santa-gift-matching/gift_goodkids_v2.csv', header=None).drop(0, axis=1).values
chi = np.full((1000000, 1000), -10,dtype=np.int16)

gif = np.full((1000000, 1000), -1,dtype=np.int16)

VAL = (np.arange(200,0,-2)+1)*10

for c in range(1000000):

    chi[c,child_prefs[c]] += VAL 

VAL = (np.arange(2000,0,-2)+1)

for g in range(1000):

    gif[gift_prefs[g],g] += VAL

    

def calc_score(ChildId,GiftId):

    return (sum(chi[ChildId,GiftId])/2000000000)**3, (sum(gif[ChildId,GiftId])/2000000000)**3
%%time

df = pd.read_csv('../input/santa-gift-matching/sample_submission_random_v2.csv')

score_chi, score_gif = calc_score(df.ChildId,df.GiftId)

print('score',score_chi,score_gif,' sum ',score_chi + score_gif)