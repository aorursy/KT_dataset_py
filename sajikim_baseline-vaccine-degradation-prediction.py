# load library

import os

import json

import numpy as np

import pandas as pd
# read train, test, sample and show shape of all.

df_train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

df_test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

df_sample = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')



df_train.shape, df_test.shape, df_sample.shape
# concat train and test.

df_train['dataset'] = 'train'

df_test['dataset'] = 'test'

df_all = pd.concat([df_train, df_test])
df_train.head()
df_test.head()
df_sample.head()
df_all.info()
# read npy data file

bpps_list = os.listdir('../input/stanford-covid-vaccine/bpps/')

bpps_npy = np.load(f'../input/stanford-covid-vaccine/bpps/{bpps_list[0]}')

print('Count of npy files: ', len(bpps_list))

print('Size of image: ', bpps_npy.shape)
# show the images of npy data.

import matplotlib.pyplot as plt

from skimage import color

from skimage import io



fig = plt.figure(figsize=(15, 15))

for i, f in enumerate(bpps_list):

    if i == 25:

        break

    sub = fig.add_subplot(5,5, i + 1)

    example_bpps = np.load(f'../input/stanford-covid-vaccine/bpps/{f}')

    sub.imshow(example_bpps,interpolation='nearest')

    sub.set_title(f)

plt.tight_layout()

plt.show()

a = []

b = []

for i, f in enumerate(bpps_list):

    test_bpps = np.load(f'../input/stanford-covid-vaccine/bpps/{f}')

    for j, test_bpp in enumerate(test_bpps):

        a += [os.path.splitext(f)[0] + "_" + str(j)]

        b += [test_bpp.mean()]

        

df_npy=pd.DataFrame(data={'id_seqpos': a, 'mean_npy': b})
df_npy['mean_npy2']=df_npy['mean_npy']/df_npy['mean_npy'].mean()
df_train['mean_reactivity'] = df_train['reactivity'].apply(lambda x: np.mean(x))

df_train['mean_deg_Mg_pH10'] = df_train['deg_Mg_pH10'].apply(lambda x: np.mean(x))

df_train['mean_deg_pH10'] = df_train['deg_pH10'].apply(lambda x: np.mean(x))

df_train['mean_deg_Mg_50C'] = df_train['deg_Mg_50C'].apply(lambda x: np.mean(x))

df_train['mean_deg_50C'] = df_train['deg_50C'].apply(lambda x: np.mean(x))



mean_react = df_train['mean_reactivity'].mean()

mean_deg_Mg_pH10 = df_train['mean_deg_Mg_pH10'].mean()

mean_deg_pH10 = df_train['mean_deg_pH10'].mean()

mean_deg_Mg_50C = df_train['mean_deg_Mg_50C'].mean()

mean_deg_50C = df_train['mean_deg_50C'].mean()



df_sample = pd.merge(df_sample, df_npy, on='id_seqpos', how='left')



df_sample['reactivity'] = mean_react

df_sample['deg_Mg_pH10'] = mean_deg_Mg_pH10

df_sample['deg_pH10'] = mean_deg_pH10

df_sample['deg_Mg_50C'] = mean_deg_Mg_50C

df_sample['deg_50C'] = mean_deg_50C



df_sample['reactivity'] = df_sample['reactivity'] * df_sample['mean_npy2'] 

df_sample['deg_Mg_pH10'] = df_sample['deg_Mg_pH10'] * df_sample['mean_npy2'] 

df_sample['deg_pH10'] = df_sample['deg_pH10'] * df_sample['mean_npy2'] 

df_sample['deg_Mg_50C'] = df_sample['deg_Mg_50C'] * df_sample['mean_npy2'] 

df_sample['deg_50C'] = df_sample['deg_50C'] * df_sample['mean_npy2'] 



df_sample.drop(columns=['mean_npy', 'mean_npy2'], inplace=True)
df_sample.to_csv('submission.csv', index=False)

df_sample.head()