import numpy as np 

import pandas as pd 
data_path = "../input/"

train_file = data_path + "train.json"

test_file = data_path + "test.json"

train_df = pd.read_json(train_file)

test_df = pd.read_json(test_file)



train_df = train_df.fillna('')

test_df = test_df.fillna('')



train_df['photos_num'] = train_df.photos.apply(lambda x: len(x))

test_df['photos_num'] = test_df.photos.apply(lambda x: len(x))



train_df['features_num'] = train_df.features.apply(lambda x: len(x))

test_df['features_num'] = test_df.features.apply(lambda x: len(x))



print('Shape of train dataset = ' + str(train_df.shape))

print('Shape of test dataset = ' + str(test_df.shape))
def find_idx_ainb(a, b, cols):

    if len(cols) == 0:

        return []

    

    ainb = a[cols[0]].isin(b[cols[0]].values)

    if len(cols) > 1:

        for i in range(1, len(cols)):

            ainb &= a[cols[i]].isin(b[cols[i]].values)

            

    return ainb
cols = ['bathrooms', 'bedrooms', 'building_id', 'description']

print(cols)

idx = find_idx_ainb(train_df, test_df, cols)

print('\nPercent of train data in test = {0:.2f}%'.format(100*np.mean(idx)))
cols = list(train_df.columns.values)

cols.remove('created')

cols.remove('features')

cols.remove('interest_level')

cols.remove('listing_id')

cols.remove('photos')

print(cols)



idx = find_idx_ainb(train_df, test_df, cols)

print('\nPercent of train data in test = {0:.2f}%'.format(100*np.mean(idx)))
cols = list(train_df.columns.values)

cols.remove('features')

cols.remove('interest_level')

cols.remove('listing_id')

cols.remove('photos')

print(cols)



idx = find_idx_ainb(train_df, test_df, cols)

print('\nPercent of train data in test = {0:.2f}%'.format(100*np.mean(idx)))
cols = ['bathrooms', 'bedrooms', 'building_id', \

        'description', 'display_address', 'latitude', \

        'longitude', 'manager_id', 'price', 'street_address', \

        'photos_num', 'features_num']

df_merged = pd.merge(train_df, test_df, \

                     on=cols, \

                     suffixes=('_train', '_test'), how='right')

df_merged = df_merged.rename(columns={'listing_id_test': 'listing_id'})

df_merged.head()
fname = 'sample_submission.csv'

subm = pd.read_csv(data_path + fname)

subm = subm.merge(df_merged[['listing_id','interest_level']], on='listing_id')
print('Number of duplicates = ' + str(np.sum(subm.duplicated(subset='listing_id'))))
subm.sort_values('listing_id').loc[subm.duplicated(subset='listing_id', keep=False)].head(10)
print('Number of duplicates in train = ' + \

      str(np.sum(train_df.duplicated(subset=cols, keep=False))))

print('Number of duplicates in test = ' + \

      str(np.sum(test_df.duplicated(subset=cols, keep=False))))
subm.low.loc[subm.interest_level=='low'] = 1.0

subm.medium.loc[subm.interest_level=='low'] = 0.0

subm.high.loc[subm.interest_level=='low'] = 0.0



subm.low.loc[subm.interest_level=='medium'] = 0.0

subm.medium.loc[subm.interest_level=='medium'] = 1.0

subm.high.loc[subm.interest_level=='medium'] = 0.0



subm.low.loc[subm.interest_level=='high'] = 0.0

subm.medium.loc[subm.interest_level=='high'] = 0.0

subm.high.loc[subm.interest_level=='high'] = 1.0



subm = subm.groupby('listing_id').mean()



print('subm.shape = ' + str(subm.shape))

subm.head()
subm.to_csv('submission.csv', index=True)