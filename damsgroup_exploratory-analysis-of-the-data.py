import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
INPUT_PATH = "../input/"

child_pref = pd.read_csv(INPUT_PATH + 'child_wishlist.csv', header=None)

santa_pref = pd.read_csv(INPUT_PATH + 'gift_goodkids.csv', header=None)

santa_pref.head()
len((pd.unique(santa_pref[list(range(1,1001))].values.ravel('K'))))

ls= [pd.value_counts(child_pref[i].values, sort = False).sort_index() for i in range(1,11)]

ar = np.zeros((1000,12))

ar[:,1:-1] = np.array(ls).T

ar[:,0] = ar.sum(axis=1) # Sum 

ar[:,11] = range(0,1000) # Column with ID's of present

ar = np.array(sorted(ar, key = lambda x: x[0])).astype(int)



print('Less interesting presents')

[print('Present ID = ', ar[i, 11],' Sum =', ar[i,0],  ' Children preference', ar[i,1:11]) for i in range(0,10)]

print('Most interesting presents')

[print('Present ID = ', ar[i, 11],' Sum =', ar[i,0],  ' Children preference', ar[i,1:11]) for i in range(990,1000)]