import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



train=pd.read_csv('../input/train_numeric.csv', usecols=['Id','Response'])

failure_ids = sorted(train[train['Response']==1]['Id'].values)

count = {}

for i in range(1, len(failure_ids)):

    d = abs(failure_ids[i] - failure_ids[i-1])

    if d not in count:

        count[d] = 0            

    count[d] = count[d] + 1



near = {k:v for k,v in count.items() if k < 10 or v > 50}

plt.bar(near.keys(), near.values())


