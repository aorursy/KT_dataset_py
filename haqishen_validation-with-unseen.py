import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold
df_train = pd.read_csv('../input/bengaliai-cv19/train.csv')
df_train.head(2)
grapheme2idx = {grapheme: idx for idx, grapheme in enumerate(df_train.grapheme.unique())}

df_train['grapheme_id'] = df_train['grapheme'].map(grapheme2idx)
df_train.head(2)
n_fold = 5

skf = StratifiedKFold(n_fold, random_state=42)

for i_fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train.grapheme)):

    df_train.loc[val_idx, 'fold'] = i_fold

df_train['fold'] = df_train['fold'].astype(int)
df_train.head(2)
df_train['unseen'] = 0

df_train.loc[df_train.grapheme_id >= 1245, 'unseen'] = 1
df_train.unseen.value_counts()
df_train.loc[df_train['unseen'] == 1, 'fold'] = -1
df_train['fold'].value_counts()
df_train.head(2)
df_train.to_csv('train_v2.csv', index=False)
n_fold = 5

for fold in range(n_fold):

    train_idx = np.where((df_train['fold'] != fold) & (df_train['unseen'] == 0))[0]

    valid_idx = np.where((df_train['fold'] == fold) | (df_train['unseen'] != 0))[0]



    df_this_train = df_train.loc[train_idx].reset_index(drop=True)

    df_this_valid = df_train.loc[valid_idx].reset_index(drop=True)

    

    #################################

    # Do training and validating here

    #################################

    

    break
n_uniq_grapheme = df_this_train.grapheme_id.nunique()

n_uniq_root = df_this_train.grapheme_root.nunique()

n_uniq_vowel = df_this_train.vowel_diacritic.nunique()

n_uniq_diacritic = df_this_train.consonant_diacritic.nunique()



print(f'We have only {n_uniq_grapheme} grapheme in training data, but all {n_uniq_root} roots, {n_uniq_vowel} vowels, {n_uniq_diacritic} diacritics are remains')
n_uniq_grapheme = df_this_valid.grapheme_id.nunique()

n_uniq_root = df_this_valid.grapheme_root.nunique()

n_uniq_vowel = df_this_valid.vowel_diacritic.nunique()

n_uniq_diacritic = df_this_valid.consonant_diacritic.nunique()



print(f'While we have all {n_uniq_grapheme} grapheme in validation, and all {n_uniq_root} roots, {n_uniq_vowel} vowels, {n_uniq_diacritic} diacritics as well')
# We have 7578 unseen samples in validation set, which is approximately 16.4%

df_this_valid['unseen'].value_counts()