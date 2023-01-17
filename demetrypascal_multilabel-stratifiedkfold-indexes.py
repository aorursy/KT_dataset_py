import numpy as np

import pandas as pd

import json

!pip install iterative-stratification
cv_count = 10



seeds = list(range(15))







files = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



files = sorted(files)



files
train = pd.read_csv(files[2])



targets = pd.read_csv(files[4])



train.head()
targets = targets[train['cp_type']!= 'ctl_vehicle']



train = train[train['cp_type']!= 'ctl_vehicle']



train.shape
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



dic = {}



for s in seeds:

    mskf = MultilabelStratifiedKFold(n_splits=cv_count, shuffle=True, random_state=s)

    seed_dic = {}

    for fold, (train_index, test_index) in enumerate(mskf.split(train, targets)):

        seed_dic[str(fold)] = {

            'train': [str(ind) for ind in train_index],

            'test': [str(ind) for ind in test_index]

        }

    dic[str(s)] = seed_dic

with open('indexes.json', 'w') as file:

    json.dump(dic, file, indent = 4)