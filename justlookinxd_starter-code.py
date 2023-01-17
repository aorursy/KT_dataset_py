import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from tqdm import tqdm_notebook as tqdm

print(os.listdir("../input"))
train = pd.read_csv('../input/recognizing-faces-in-the-wild/train_relationships.csv')

sample_submission = pd.read_csv('../input/recognizing-faces-in-the-wild/sample_submission.csv')
train.shape , sample_submission.shape
train.describe()
train.head()
print(os.listdir("../input/recognizing-faces-in-the-wild/train/")[:17])
l =[]

for fam in tqdm(os.listdir("../input/recognizing-faces-in-the-wild/train/")):

    for person in os.listdir("../input/recognizing-faces-in-the-wild/train/"+fam):

        l.append(fam+'/'+person)
len(l)
from itertools import product
matrix = []

while(len(l)!=1):

    p = l[0]

    l.remove(p)

    matrix.append(np.array(list(product([p],l)), dtype='object'))

matrix = pd.DataFrame(np.vstack(matrix), columns=['p1','p2'])
matrix.head()
train['target']=1

matrix = pd.merge(matrix, train, on=['p1','p2'], how='left')
matrix['target'] = (matrix['target']

                            .fillna(0)

                            .astype(np.int8))
matrix.head(15)
matrix.target.value_counts()