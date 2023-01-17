# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_v2.csv')

test = pd.read_csv('../input/test_v2_file_mapping.csv')
# mapping labels to integer classes

flatten = lambda l: [item for sublist in l for item in sublist]

labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}

inv_label_map = {i: l for l, i in label_map.items()}
y_train = []

# labels for the train dataset

for f, tags in tqdm(train.values, miniters=1000):

    targets = np.zeros(17)

    for t in tags.split(' '):

        targets[label_map[t]] = 1 

    y_train.append(targets)

    

y_train = np.array(y_train, np.uint8)
trn_index = []

val_index = []

# change split value for getting different validation splits

split = .2

index = np.arange(len(train))

for i in tqdm(range(0,17)):

    sss = StratifiedShuffleSplit(n_splits=2, test_size=split, random_state=i)

    for train_index, test_index in sss.split(index,y_train[:,i]):

        X_train, X_test = index[train_index], index[test_index]

    # to ensure there is no repetetion within each split and between the splits

    trn_index = trn_index + list(set(list(X_train)) - set(trn_index) - set(val_index))

    val_index = val_index + list(set(list(X_test)) - set(val_index) - set(trn_index))

 
len(trn_index), len(val_index)
dist = []

#checking distribution for each class

for i in range(17):

    dist.append(np.unique(y_train[trn_index,i],return_counts=True)[1][1]/len(trn_index))

    dist.append(np.unique(y_train[val_index,i],return_counts=True)[1][1]/len(val_index))

dist_label = [x for pair in zip([x + '_trn' for x in (inv_label_map.values())],

                                [x + '_val' for x in (inv_label_map.values())]) for x in pair]
plot = plt.figure(figsize=(10,10))

plt.barh(np.arange(len(dist)),dist)

plt.yticks(np.arange(len(dist)),dist_label,rotation = 45)

plt.grid(True)

plt.show()