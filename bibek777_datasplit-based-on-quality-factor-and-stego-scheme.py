import os

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
!ls ../input/alaska2-image-steganalysis
!ls ../input/alaska2trainvalsplit
split_dir = '../input/alaska2trainvalsplit'
train_split = pd.read_csv(f'{split_dir}/alaska2_train_df.csv')

train_split.head()
def transform(x):

    split = x.split('/')

    path = split[-2] + '/' + split[-1]

    return path
train_split['ImageFileName'] = train_split['ImageFileName'].transform(transform)

train_split.head()
train_split['Label'].value_counts()
valid_split = pd.read_csv(f'{split_dir}/alaska2_val_df.csv')

valid_split.head()
valid_split['ImageFileName'] = valid_split['ImageFileName'].transform(transform)

valid_split.head()
# combine train and valid split

df_all = pd.concat([train_split, valid_split])

df_all.head()
# sanity check that train_split + valid_split = combined 

train_split.shape[0] + valid_split.shape[0] == df_all.shape[0]
def transform(x):

    split = x.split('/')[-2]

    path = split[-2] + '/' + split[-1]

    return path
# make a column representing the stego-scheme

df_all['Stego'] = df_all['ImageFileName'].transform(lambda x: x.split('/')[-2])
df_all.head()
df_all['Stego'].value_counts()
def qf_transform(x):

    if x in [1,4,7]:

        return 75

    elif x in [2,5,8]:

        return 90

    elif x in [3,6,9]:

        return 95

    else:

        return x
# make a column representing the quality factors; for Cover images, I've set quality factor = 0

df_all['quality_factor'] = df_all['Label'].transform(qf_transform)
df_all.head()
df_all['quality_factor'].value_counts()
# save the combined df 

df_all.to_csv('df_all.csv', index=False)
# split the data based on quality factor of 75, 90 and 95

for qf in [75, 90, 95]:

    df_qf = df_all[df_all['quality_factor']==qf]

    df_qf_tr, df_qf_val_test = train_test_split(df_qf, test_size=0.3, random_state=1234, stratify=df_qf['Label'].values)

    df_qf_val, df_qf_test  = train_test_split(df_qf_val_test, test_size=0.2, random_state=1234, stratify=df_qf_val_test['Label'].values)

    print(f'Split for quality factor of {qf}...')

    #print(df_qf_tr['Label'].value_counts())

    #print(df_qf_val['Label'].value_counts())

    #print(df_qf_test['Label'].value_counts())

    print('Shape of train split: ', df_qf_tr.shape)

    print('Shape of valid split: ', df_qf_val.shape)

    print('Shape of val_test split: ', df_qf_test.shape)

    print('*'*35)

    

    #save the splits

    df_qf_tr.to_csv(f'train_split_qf_{qf}.csv', index=False)

    df_qf_val.to_csv(f'valid_split_qf_{qf}.csv', index=False)

    df_qf_test.to_csv(f'test_val_split_qf_{qf}.csv', index=False)

    
# split the data based on stego scheme of JMiPOD, UERD, JUNIWARD

for stego in ['JMiPOD', 'UERD', 'JUNIWARD']:

    df_stego = df_all[df_all['Stego']==stego]

    df_stego_tr, df_stego_val_test = train_test_split(df_stego, test_size=0.3, random_state=1234, stratify=df_stego['Label'].values)

    df_stego_val, df_stego_test  = train_test_split(df_stego_val_test, test_size=0.2, random_state=1234, stratify=df_stego_val_test['Label'].values)

    print(f'Split for Stego type {stego}...')

    #print(df_stego_tr['Label'].value_counts())

    #print(df_stego_val['Label'].value_counts())

    #print(df_stego_test['Label'].value_counts())

    print('Shape of train split: ', df_stego_tr.shape)

    print('Shape of valid split: ', df_stego_val.shape)

    print('Shape of val_test split: ', df_stego_test.shape)

    print('*'*35)

    

    #save the splits

    df_stego_tr.to_csv(f'train_split_stego_{stego}.csv', index=False)

    df_stego_val.to_csv(f'valid_split_stego_{stego}.csv', index=False)

    df_stego_test.to_csv(f'test_val_split_stego_{stego}.csv', index=False)

    