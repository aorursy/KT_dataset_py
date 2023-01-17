# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix

from sklearn.linear_model import LogisticRegressionCV

from tensorflow import keras

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.preprocessing import minmax_scale

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')
targets = train.target

train.drop('target', axis=1 , inplace=True)
all_data = pd.concat([train, test]).reset_index(drop=True)

all_data.head()
bin_cols = [bin_col for bin_col in all_data.columns if bin_col.startswith('bin')]

all_data['bin_3'] = all_data.bin_3.apply(lambda x: 1 if x == 'T' else 0)

all_data['bin_4'] = all_data.bin_4.apply(lambda x: 1 if x == 'Y' else 0)

all_data[bin_cols] = all_data[bin_cols].astype(np.uint8)
ord_cols = [ord_col for ord_col in all_data.columns if ord_col.startswith('ord')]



#ord_1

all_data['ord_1'] = all_data.ord_1.map({'Novice': 0, 'Contributor': 1, 'Expert': 3, 'Master': 4, 'Grandmaster': 5}).astype(np.uint8)



#ord_2

all_data['ord_2'] = all_data.ord_2.map({'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5}).astype(np.uint8)



#ord_3

def ord_mapper(col, ord_map={}, df=all_data, i=0):

    sorted_uniques = df[col].sort_values().unique()

    for item in sorted_uniques:

        ord_map[item] = i

        i += 1

    return ord_map



all_data['ord_3'] = all_data.ord_3.map(ord_mapper('ord_3')).astype(np.uint8)



#ord_4

all_data['ord_4'] = all_data.ord_4.map(ord_mapper('ord_4')).astype(np.uint8)



#ord_5

all_data['ord_5'] = all_data.ord_5.map(ord_mapper('ord_5')).astype(np.uint8)
# Normalize these ordinals

all_data[ord_cols] = minmax_scale(all_data[ord_cols])
# changing month and day to cyclical feature

all_data['day_sin'] = np.sin((all_data.day-1)*(2.*np.pi/30))

all_data['day_cos'] = np.cos((all_data.day-1)*(2.*np.pi/30))

all_data['month_sin'] = np.sin((all_data.month-1)*(2.*np.pi/12))

all_data['month_cos'] = np.cos((all_data.month-1)*(2.*np.pi/12))



# drop ordinal ones

all_data.drop(['day', 'month'], axis=1, inplace=True)
nom_cols = [nom_col for nom_col in all_data.columns if nom_col.startswith('nom')]

sparse_nom = pd.get_dummies(all_data[nom_cols], drop_first=True, sparse=True)

all_data.drop(nom_cols, axis=1, inplace=True)
print(all_data.shape, sparse_nom.shape)

# create a sparse csr matrix

sparse_matrix = sparse_nom.to_sparse().to_coo().tocsr()



# selecting top 7.5% effective features using chi2

k = int(sparse_nom.shape[1] * 0.075)

kbest_chi_nom_features_selector = SelectKBest(chi2, k=k).fit(sparse_matrix[:len(train)], targets)

nom_features_mask = kbest_chi_nom_features_selector.get_support()

nom_features_mask[:22] = True # Force dummies of nom_1 to nom_4 to be included as they are only 22 column

sparse_nom_selected = sparse_nom.iloc[:, nom_features_mask]
all_data = all_data.join(sparse_nom_selected.to_dense().astype(np.uint8))

train = all_data.iloc[:len(train)].set_index('id')

test = all_data.iloc[len(train):].set_index('id')
# This will take a long time..

clf = LogisticRegressionCV(Cs=50, cv=11, verbose=0, n_jobs=4, scoring='roc_auc', 

                           solver='sag', max_iter=200, tol=0.001)

clf.fit(train, targets)
pd.DataFrame({"id": test.index, "target": clf.predict_proba(test)[:, 1]}).to_csv('submission.csv', index=False)