import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import clock

from tqdm import tqdm

from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))
%%time

child_prefs = pd.read_csv('../input/santa-gift-matching/child_wishlist.csv', header=None).drop(0, axis=1).values

gift_prefs = pd.read_csv('../input/santa-gift-matching/gift_goodkids.csv', header=None).drop(0, axis=1).values

# load sample sub

df = pd.read_csv('../input/santa-gift-matching/sample_submission_random.csv')

df2 = pd.read_csv('../input/085933376csv/0.85933376.csv')
%%time

# BUILD 2G lookup table

chigif = np.full((1000000, 1000), -101,dtype=np.int16)

VAL = (np.arange(20,0,-2)+1)*100

for c in tqdm(range(1000000)):

    chigif[c,child_prefs[c]] += VAL 

VAL = (np.arange(2000,0,-2)+1)

for g in tqdm(range(1000)):

    chigif[gift_prefs[g],g] += VAL 
%%time

# COMPUTE SCORE sample sub

score = np.sum(chigif[df.ChildId,df.GiftId])/2000000000

print('score',score)
%%time

# COMPUTE SCORE own sub

score = np.sum(chigif[df2.ChildId,df2.GiftId])/2000000000

print('score',score)