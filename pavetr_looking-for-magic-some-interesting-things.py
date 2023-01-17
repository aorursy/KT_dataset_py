from tqdm import tqdm

from sklearn.preprocessing import scale

from sklearn.metrics import roc_auc_score
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
random_state = 42

np.random.seed(random_state)



train = pd.read_csv(r'../input/train.csv')

test = pd.read_csv(r'../input/test.csv')
cols = [t for t in train.columns if 'var' in t]
def little_minimizer(series_pos, series_neg):

    max_pos = series_pos.max()

    min_pos = series_pos.min()

    

    max_neg = series_neg.max()

    min_neg = series_neg.min()

    

    return [[min_pos, max_pos], [min_neg, max_neg]]



def create_minimizer(df, teach_cols, target_column = 'target'):

    dicty = {}

    for c in teach_cols:

        dicty[c] = little_minimizer(df[df[target_column] == 1][c], df[df[target_column] == 0][c])

    return dicty



def minimize_features(array, min_dict, cols, target):

    counter = 0

    array = array.tolist()

    for i, a in enumerate(array):

        if a < min_dict[cols[i]][1 - target][0] or a > min_dict[cols[i]][1 - target][1]:

            counter += 1

    return counter
minimizer = create_minimizer(train, cols, target_column = 'target')

train['var_positive'] = train[cols].apply(lambda x: minimize_features(x, minimizer, cols, 1), axis = 1)
train.groupby('var_positive')['target'].agg(['mean', 'count'])
train = train.sort_values(by = 'target', ascending = False)

for c in tqdm(range(len(cols))):

    train[cols[c]] = pd.concat([

                            train[train['target'] == 1][cols[c]].sort_values(ascending = False),

                            train[train['target'] == 0][cols[c]].sort_values(ascending = False)

                         ]).values
m1 = [81, 139, 12, 146, 76, 174, 21, 80, 166, 165, 13, 148, 198, 34, 115, 109, 44, 169, 149, 92, 108, 154, 33, 9, 192, 122, 121, 86, 123, 107, 127, 36, 172, 75, 177, 197, 87, 56, 93, 188, 131, 186, 141, 43, 104, 150, 31, 132, 23, 114, 58, 28, 116, 85, 194, 83]

m2 = [6, 110, 53, 26, 22, 99, 190, 2, 133, 0, 179, 1, 40, 184, 170, 78, 191, 94, 67, 18, 173, 118, 164, 89, 91, 147, 95, 35, 155, 106, 71, 157, 48, 162, 180, 163, 5, 145, 119, 32, 130, 49, 167, 90, 24, 195, 135, 151, 125, 128, 111, 52, 137, 70, 105, 51, 112, 199, 66, 82, 196, 175, 11, 74, 144, 8]

s = [26, 81, 139, 110, 12, 2, 22, 80, 53, 146, 179, 198, 99, 44, 0, 174, 76, 6, 166, 148, 133, 191, 40, 109, 190, 13, 123, 170, 165, 86, 108, 94, 21, 78, 1, 154, 184, 163, 91, 95, 75, 18, 93, 157, 89, 34, 119, 180, 115, 164, 92, 155, 9, 147, 56, 188, 122, 33, 130, 169, 5, 135, 51, 125, 141, 106, 151, 197, 162, 195, 172, 127, 121, 67, 111, 177, 173, 145, 132, 32, 43, 114, 131, 49, 36, 167, 88, 35, 107, 87, 175, 83, 149, 118, 196, 168, 150]
x_train = scale(train.iloc[:, 2:])
train["prediction"] = np.std(x_train[:, s], axis=1) + np.mean(x_train[:, m2], axis=1) - np.mean(x_train[:, m1], axis=1)
roc_auc_score(train['target'], train['prediction'])