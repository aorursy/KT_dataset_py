# import python standard library

import os



# import data manipulation library

import numpy as np

import pandas as pd



# import data visualization library

from tqdm import tqdm
def csvload(file: str, nrows: int = None) -> pd.DataFrame:

    """ Return a loaded csv file. """

    

    return pd.read_csv('../input/train_simplified/' + file, nrows=nrows)
# class files and dictionary

files = sorted(os.listdir('../input/train_simplified/'), reverse=False)

class_dict = {file[:-4].replace(" ", "_"): i for i, file in enumerate(files)}



# data dimensions

num_shuffles = 100
# acquiring training and testing data

for i, file in tqdm(enumerate(files)):

    df_data = csvload(file, nrows=30000)

    df_data['shuffle'] = (df_data['key_id'] // 10 ** 7) % num_shuffles

    for k in range(num_shuffles):

        df_chunk = df_data[df_data['shuffle'] == k]

        if i == 0: df_chunk.to_csv('train_k%d.csv' %k, index=False)

        else: df_chunk.to_csv('train_k%d.csv' %k, header=False, index=False, mode='a')            
# shuffle and compress file

for k in tqdm(range(num_shuffles)):

    df_data = pd.read_csv('train_k%d.csv' %k)

    print(df_data.shape)

    df_data['rand'] = np.random.rand(df_data.shape[0])

    df_data = df_data.sort_values(['rand']).drop(['rand'], axis=1)

    df_data.to_csv('train_k%d.csv.gz' %k, index=False, compression='gzip')

    

    # memory clean-up

    os.remove('train_k%d.csv' %k)