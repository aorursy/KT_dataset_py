import numpy as np
import pandas as pd
import networkx as nx
from networkx.utils import cuthill_mckee_ordering
gibasfeatures = ["f190486d6","58e2e02e6","eeb9cd3aa","9fd594eec","6eef030c1",
                 "15ace8c9f","fb0f5dbfe","58e056e12","20aa07010","024c577b9",
                 "d6bb78916","b43a7cfd5","58232a6fb"]
train = pd.read_csv('../input/train.csv')
orig = train[train.columns[2:]].copy()
orig.head()
G = nx.from_numpy_matrix((orig.values[:,:4459])) #Make it square ignore last columns and shut eyes symmetry wise
cm = list(cuthill_mckee_ordering(G))
A = nx.adjacency_matrix(G, nodelist=cm)
cm_df = pd.DataFrame(A.todense(),columns = np.array(orig.columns)[cm], index=orig.index[cm])
cm_df.head()
featurecount = cm_df[cm_df!=0].count(axis=0)
featurecount.head()
for f in gibasfeatures:
    print(featurecount[featurecount.index==f])
featurecount[featurecount>=1447].shape #Only 56 features higher than 1447
features = featurecount[featurecount>=1447].index.ravel()
set(gibasfeatures).intersection(features)