import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
train_set = pd.read_csv('../input/training_set.csv')
train_md = pd.read_csv('../input/training_set_metadata.csv')
train_set.head()
train_md.head()
class_ids = train_md['target'].unique()
class_ids.sort(axis=0)
oids = [(c, train_md[train_md['target']==c].sample(n=3, random_state=2018)['object_id'].values) for c in class_ids]
oids
for cid, oid in oids:
    plt.figure(figsize=(10, 10))
    for band in range(6):        
        plt.subplot(6, 1, band+1)
        if band == 0:
            plt.title('Target {} (object_id = {})'.format(cid, oid))
        plt.ylabel('Band {}'.format(band))
        for i in range(3):
            ts = train_set[((train_set['object_id']==oid[i]) & (train_set['passband']==band))]
            plt.plot(ts['mjd'], ts['flux'], 'x-')
    plt.show()
    