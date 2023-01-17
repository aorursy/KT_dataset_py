import os

import numpy as np

import pandas as pd



import random

from collections import Counter, defaultdict



SEED = 1234
# https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation



def stratified_group_k_fold(X, y, groups, k, seed=None):

    labels_num = np.max(y) + 1

    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))

    y_distr = Counter()

    for label, g in zip(y, groups):

        y_counts_per_group[g][label] += 1

        y_distr[label] += 1



    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))

    groups_per_fold = defaultdict(set)



    def eval_y_counts_per_fold(y_counts, fold):

        y_counts_per_fold[fold] += y_counts

        std_per_label = []

        for label in range(labels_num):

            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])

            std_per_label.append(label_std)

        y_counts_per_fold[fold] -= y_counts

        return np.mean(std_per_label)

    

    groups_and_y_counts = list(y_counts_per_group.items())

    random.Random(seed).shuffle(groups_and_y_counts)



    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):

        best_fold = None

        min_eval = None

        for i in range(k):

            fold_eval = eval_y_counts_per_fold(y_counts, i)

            if min_eval is None or fold_eval < min_eval:

                min_eval = fold_eval

                best_fold = i

        y_counts_per_fold[best_fold] += y_counts

        groups_per_fold[best_fold].add(g)



    all_groups = set(groups)

    for i in range(k):

        train_groups = all_groups - groups_per_fold[i]

        test_groups = groups_per_fold[i]



        train_indices = [i for i, g in enumerate(groups) if g in train_groups]

        test_indices = [i for i, g in enumerate(groups) if g in test_groups]



        yield train_indices, test_indices
path = '../input/understanding_cloud_organization'

os.listdir(path)
n_train = len(os.listdir(f'{path}/train_images'))

n_test = len(os.listdir(f'{path}/test_images'))

print( f'There are {n_train} images in train dataset' )

print( f'There are {n_test} images in test dataset' )
train_df = pd.read_csv(f'{path}/train.csv')
train_df.head()
train_df['class_name'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])

train_df['image_name'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
# https://www.kaggle.com/bibek777/5fold-stratified-split



train_df['exists'] = train_df['EncodedPixels'].notnull().astype(int)



class_names_dict = {'Fish':1, 'Flower':2, 'Gravel':3, 'Sugar':4}

train_df['class_id'] = train_df['class_name'].map(class_names_dict)

train_df['class_id'] = [row.class_id if row.exists else 0 for row in train_df.itertuples()]



train_df = train_df.sort_values( by='Image_Label', ascending=True )

train_df.reset_index(drop=True, inplace=True)
train_df.head()
groups = train_df.image_name.values

labels = train_df.class_id.values



splits = list( stratified_group_k_fold( train_df, labels, groups, k=5, seed=SEED ) )
for i, (train_idx, valid_idx) in enumerate(splits):

    

    train = train_df.iloc[train_idx, :]  

    valid = train_df.iloc[valid_idx, :]

    

    train_ids = train['image_name'].drop_duplicates().values

    valid_ids = valid['image_name'].drop_duplicates().values    

    

    print( '=========================' )

    print( 'K=', i+1 )

    

    print( '[train]' )

    print( 'class: ', Counter(train['class_id']) )   

    print( 'images: ', len(train_ids) )   

    

    print( '===' )

    

    print( '[valid]' )

    print( 'class: ', Counter(valid['class_id']) )

    print( 'images: ', len(valid_ids) )   