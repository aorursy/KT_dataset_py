import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

cols=train.columns.drop(['id','target']).tolist()
from itertools import combinations

lt=len(test)

ltt=len(train)

for c in combinations(cols,10):

    print(c)

    lt1 =len(test.drop_duplicates(c))

    ltt1=len(train.drop_duplicates(c))

    print('    train:',ltt1,'/',ltt,' (',ltt1/ltt*100,'%)')

    print('    test:' ,lt1 ,'/',lt ,' (',lt1 /lt *100,'%)')
