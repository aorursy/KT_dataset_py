import os

import numpy as np

import pandas as pd

from pathlib import Path



pd.set_option('max_columns', None)

pd.set_option('max_rows', None)
show_files=0



if show_files:

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

dups=pd.read_csv('/kaggle/input/siim-list-of-duplicates/2020_Challenge_duplicates.csv')



# TRAIN_IMG=Path('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/')

# TEST_IMG=Path('/kaggle/input/siim-isic-melanoma-classification/jpeg/test/')
train.head()
dups.head()
len(dups)
unq=np.unique(dups['ISIC_id'].values)

len(unq)
def extract_data(partition, verbose=1):

    

    print(f"Partition: {partition}")

    

    if partition=='train':

        df=train.copy()

    else:

        df=test.copy()

        

    mask_df=dups['partition']==partition



    

    original=dups[mask_df].merge(df, left_on='ISIC_id', right_on='image_name')



        

    paired=dups[mask_df].merge(df, left_on='ISIC_id_paired', right_on='image_name')

    

    if verbose:

        print(f"The total number of entries: {mask_df.sum()}")    

        print(f"The length of 'original': {len(original)}")    

        print(f"The length of 'paired': {len(paired)}")

    

    return original, paired, df
original_train, paired_train, _ = extract_data('train')
original_train.head()
def check_consistency(partition, verbose=0):

    

    original, paired, df = extract_data(partition, verbose=verbose)

    

    cols=[c for c in df.columns if c not in ['image_name']]



    for c in cols:

        print("="*100)

        print(f"{c}:")

        mask_c=np.equal(original[c].fillna('na').values, paired[c].fillna('na').values)

        if mask_c.all():

            print(f"The values of '{c}' are in a perfect agreement between the original and paired images.")

        else:

            print(f"The values of {c} differ between the original and paired images.\n")

            df_cols=['ISIC_id', 'ISIC_id_paired', c]

            df=original.loc[~mask_c, df_cols]

            df[c+'_o']=df.pop(c)

            df[c+'_p']=paired.loc[~mask_c, c]

            print(df)

    print("="*100)
check_consistency('train')
check_consistency('test')