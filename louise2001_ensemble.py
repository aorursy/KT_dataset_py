import numpy as np

import pandas as pd

from scipy.stats import rankdata

import os

from copy import deepcopy as dc



f = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')[['image_name']]

n = f.shape[0]

cols = {}

files = [os.path.join('/kaggle/input/melanoma', file) for file in os.listdir('/kaggle/input/melanoma')] + [os.path.join('/kaggle/input/melanoma-public', file) for file in os.listdir('/kaggle/input/melanoma-public')]

for i, filename in enumerate(files):

    cols[filename] = f'target_{i}'

    ff = pd.read_csv(filename)

    ff.columns = ['image_name', f'target_{i}']

    ff[f'target_{i}'] = rankdata(ff[f'target_{i}'].values.tolist())

    ff[f'target_{i}'] = ff[f'target_{i}'].apply(lambda x: (x-1)/(n-1))

    f = f.merge(ff, on='image_name')
cols
f.head()
print(f.shape)
f['target'] = f[cols.values()].mean(axis=1)

f[['image_name', 'target']].head()
f[['image_name', 'target']].to_csv('submission.csv', index=False)
def calculate_mse(y, df):

    for col in cols.values():

        df[col] = (df[col]-y)**2

    return df.sum(axis=1)

dic_errors = {}

for file, col in cols.items():

    y = f[col]

    error = calculate_mse(y,f[cols.values()])

    dic_errors[file] = error
err_df = pd.DataFrame(dic_errors).transpose().sum(axis=1).sort_values(ascending=False)

err_df
N=4
biggest_error = err_df.index.tolist()[:N]
biggest_error
f[f'target_wo_{N}'] = f[[c for k,c in cols.items() if k not in biggest_error]].mean(axis=1)

f[['image_name', f'target_wo_{N}']].to_csv(f'sub_wo_{N}.csv', index=False, header=['image_name', 'target'])

name_col = f'target_wo_{N}'
min_dist = pd.DataFrame(dic_errors).idxmin(axis=1)

min_vals = []

for i, sub in min_dist.iteritems():

    min_vals.append(f.loc[i,cols[sub]])
f['target_arg_min'] = min_vals
f[['image_name', 'target_arg_min']].to_csv('sub_argmin.csv', index=False, header=['image_name', 'target'])
vals = []

for i, row in pd.DataFrame(dic_errors).iterrows():

    vals.append(f.loc[i,[cols[sub] for sub in row.sort_values(ascending=False).index.tolist()[N+1:]]].mean())

f['target_mean_min'] = vals

f[['image_name', 'target_mean_min']].to_csv('sub_mean_min.csv', index=False, header=['image_name', 'target'])
f['global_sub']= f[[name_col, 'target_mean_min']].mean(axis=1)

f[['image_name', 'global_sub']].to_csv('global_sub.csv', index=False, header=['image_name', 'target'])