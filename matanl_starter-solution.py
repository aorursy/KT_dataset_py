import pandas as pd

import numpy as np

import sys

import os

import sklearn

from collections import namedtuple

import pandas as pd

import hashlib
x1_prefix = 'x1_'

x2_prefix = 'x2_'

similarity_label = 'same_user'

sys.path.append('../input')

sys.path.append('.')
def read_datasets():

    test_set_path = '../input/test_set/test_set.csv'

    train_set_path = '../input/train_data/train_data.csv'

    test = pd.read_csv(test_set_path,low_memory=False)

    train = pd.read_csv(train_set_path,low_memory=False)

    return train, test



def calc_similarity(test_set, fingerprint_to_hash_func):

    """

    Takes the test set and a function (pointer to your hash function)

    Returns a dataframe with ID and Prediction columns.

    """

    

    ColumnsDescriptor = namedtuple('ColumnDesc', ['hash_col_name', 'columns_list'])

    fingerprint_required_num_of_values = 778

    number_of_columns = len(test_set.columns)

    print(f'The dataset has {number_of_columns} columns')



    x1_columns = [col for col in test_set.columns if x1_prefix in col]

    x2_columns = [col for col in test_set.columns if x2_prefix in col]



    res_for_kaggle = pd.DataFrame(test_set['Id'])

    res_for_kaggle_no_hash = pd.DataFrame()



    x1_desc = ColumnsDescriptor(f'{x1_prefix}hash', x1_columns)

    x2_desc = ColumnsDescriptor(f'{x2_prefix}hash', x2_columns)

    

    for columns_set in [x1_desc, x2_desc]:

        single_sample_df = test_data[columns_set.columns_list]

        res_for_kaggle[columns_set.hash_col_name] = single_sample_df.apply(create_fingerprint_string, axis=1)



    res_for_kaggle['Predicted'] = res_for_kaggle[x1_desc.hash_col_name] == res_for_kaggle[x2_desc.hash_col_name] #create boolean results

    res_for_kaggle['Predicted'] = res_for_kaggle['Predicted'] #convert the prediction to float

    res_for_kaggle = res_for_kaggle[['Id','Predicted']]

    return res_for_kaggle

    

train_data, test_data = read_datasets()
def create_fingerprint_string(single_fingerprint):

    """

    The function accept a Series that represents a single fingerprint.

    :param single_fingerprint: an array of values of the fingerprint

    :return:

    """

    single_fingerprint.index = [col[3:] for col in single_fingerprint.index]  # remove prefix from the column name e.g x1_

    selected_columns_df = single_fingerprint.loc[["Screen.result.availWidth", "Screen.result.width"]] # This is a simple and weak example! Feature selection is not a good solution and will not work well in the real world, where feature's importance changes.

    hashed_columns = ','.join([str(x) for x in selected_columns_df.values])

    hash_str = do_hash(str(hashed_columns))

    return hash_str



def do_hash(value):

    hash_func = hashlib.md5()

    hash_func.update(str(value).encode('utf-8'))

    return hash_func.digest()
submission_data = calc_similarity(test_data, fingerprint_to_hash_func=create_fingerprint_string)
result_file_name = 'submission.csv'

submission_data.to_csv(result_file_name,index=False) 