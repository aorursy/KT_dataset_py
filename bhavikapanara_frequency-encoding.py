import pandas as pd

import numpy as np
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



train.shape,test.shape
train.head()
enc_nom_1 = (train.groupby('nom_1').size()) / len(train)

enc_nom_1
train['nom_1_encode'] = train['nom_1'].apply(lambda x : enc_nom_1[x])
train.head()