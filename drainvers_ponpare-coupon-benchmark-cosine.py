import os



def list_all_files_in(dirpath):

    for dirname, _, filenames in os.walk(dirpath):

        for filename in filenames:

            print(os.path.join(dirname, filename))



list_all_files_in('../input')
# Dataframes

import pandas as pd



# Linear algebra

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# List shifting

from collections import deque



# Sparse matrices

from scipy import sparse



# Implicit

import implicit



# Displaying stuff

from IPython.display import display



# ZIP I/O

import zipfile



# Paths

from pathlib import Path



# Timing

import time



# Disable warnings

import warnings; warnings.simplefilter('ignore')
df_coupon_list_train = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_list_train_translated.csv')

# df_coupon_area_train = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_area_train_translated.csv')

df_coupon_detail_train = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_detail_train_translated.csv')

# df_coupon_visit_train = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_visit_train_translated.csv')



df_coupon_list_test = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_list_test_translated.csv')

# df_coupon_area_test = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_area_test_translated.csv')



df_user_list = pd.read_csv('../input/coupon-purchase-prediction-translated/user_list_translated.csv')

# df_prefecture_locations = pd.read_csv('../input/coupon-purchase-prediction-translated/prefecture_locations_translated.csv')

# df_submission = pd.read_csv('../input/coupon-purchase-prediction-translated/sample_submission.csv')
df_purchased_coupons_train = df_coupon_detail_train.merge(df_coupon_list_train, on='COUPON_ID_hash', how='inner')
features = ['COUPON_ID_hash', 'USER_ID_hash', 'GENRE_NAME', 'DISCOUNT_PRICE', 'large_area_name', 'ken_name', 'small_area_name']

df_purchased_coupons_train = df_purchased_coupons_train[features]

df_purchased_coupons_train
df_coupon_list_test['USER_ID_hash'] = 'dummyuser'

df_coupon_list_test = df_coupon_list_test[features]

df_coupon_list_test
df_combined = pd.concat([df_purchased_coupons_train, df_coupon_list_test], axis=0)

df_combined['DISCOUNT_PRICE'] = 1 / np.log10(df_combined['DISCOUNT_PRICE'])

features.extend(['DISCOUNT_PRICE'])

df_combined
categoricals = ['GENRE_NAME', 'large_area_name', 'ken_name', 'small_area_name']

df_combined_categoricals = df_combined[categoricals]

df_combined_categoricals = pd.get_dummies(df_combined_categoricals, dummy_na=False)

df_combined_categoricals
continuous = list(set(features) - set(categoricals))

df_combined = pd.concat([df_combined[continuous], df_combined_categoricals], axis=1)

print(df_combined.isna().sum())

NAN_SUBSTITUTION_VALUE = 1

df_combined = df_combined.fillna(NAN_SUBSTITUTION_VALUE)

df_combined
df_train = df_combined[df_combined['USER_ID_hash'] != 'dummyuser']

df_test = df_combined[df_combined['USER_ID_hash'] == 'dummyuser']

df_test = df_test.drop('USER_ID_hash', axis=1)

display(df_train)

display(df_test)
df_train_dropped_coupons = df_train.drop('COUPON_ID_hash', axis=1)

df_user_profiles = df_train_dropped_coupons.groupby('USER_ID_hash').mean()

df_user_profiles
FEATURE_WEIGHTS = {

    'GENRE_NAME': 2,

    'DISCOUNT_PRICE': 2,

    'large_area_name': 0.5,

    'ken_name': 1.5,

    'small_area_name': 5

}
def find_appropriate_weight(weights_dict, colname):

    for col, weight in weights_dict.items():

        if col in colname:

            return weight

    raise ValueError
W_values = [find_appropriate_weight(FEATURE_WEIGHTS, colname)

            for colname in df_user_profiles.columns]

W = np.diag(W_values)

W
df_test_only_features = df_test.drop('COUPON_ID_hash', axis=1)

similarity_scores = np.dot(np.dot(df_user_profiles, W), df_test_only_features.T)

similarity_scores
s_coupons_ids = df_test['COUPON_ID_hash']

index = df_user_profiles.index

columns = pd.Series([s_coupons_ids[i] for i in range(0, similarity_scores.shape[1])], name='COUPON_ID_hash')

df_results = pd.DataFrame(index=index, columns=columns, data=similarity_scores)

df_results
def get_top10_coupon_hashes_string(row):

    sorted_row = row.sort_values()

    return ' '.join(sorted_row.index[-10:][::-1].tolist())
output = df_results.apply(get_top10_coupon_hashes_string, axis=1)

output
df_output = pd.DataFrame(data={'USER_ID_hash': output.index, 'PURCHASED_COUPONS': output.values})

df_output
df_output_all = pd.merge(df_user_list, df_output, how='left', on='USER_ID_hash')

df_output_all.to_csv('cosine_sim_python.csv', header=True, index=False, columns=['USER_ID_hash', 'PURCHASED_COUPONS'])

df_output_all