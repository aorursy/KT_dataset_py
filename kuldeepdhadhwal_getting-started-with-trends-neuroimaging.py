import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nibabel as nib

from tqdm import tqdm

from scipy import stats

import matplotlib

import matplotlib.pyplot as plt # for plotting

import seaborn as sns
ROOT = "/kaggle/input/trends-assessment-prediction/"

!ls {ROOT}
# image and mask directories

data_dir = f'{ROOT}/fMRI_train'

loading_data = f'{ROOT}loading.csv'

icn_data = f'{ROOT}ICN_numbers.csv'

fnc_data = f'{ROOT}fnc.csv'

train_scores_data = f'{ROOT}train_scores.csv'

fmri_mask = f'{ROOT}fMRI_mask.nii'
load_data = pd.read_csv(loading_data)

icn_data = pd.read_csv(icn_data)

fnc_data = pd.read_csv(fnc_data)

train_scores_data = pd.read_csv(train_scores_data)
load_data.head()
print(load_data.shape)
load_data.head()
icn_data.head()
icn_data.shape
fnc_data.head()
train_scores_data.head()
targets= load_data.columns[1:]

fig, axes = plt.subplots(6, 5, figsize=(18, 15))

axes = axes.ravel()

bins = np.linspace(-0.05, 0.05, 20)



for i, col in enumerate(targets):

    ax = axes[i]

    sns.distplot(load_data[col], label=col, kde=False, bins=bins, ax=ax)

plt.tight_layout()

plt.show()

plt.close()
targets= train_scores_data.columns[1:]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))

axes = axes.ravel()

bins = np.linspace(0, 100, 20)



for i, col in enumerate(targets):

    ax = axes[i]

    sns.distplot(train_scores_data[col], label=col, kde=False, bins=bins, ax=ax)



plt.tight_layout()

plt.show()

plt.close()
fig, ax = plt.subplots(figsize=(8, 6))

cols = load_data.columns[1:]

sns.heatmap(load_data[cols].corr(), ax=ax)
fig, ax = plt.subplots(figsize=(8, 6))

cols = train_scores_data.columns[1:]

sns.heatmap(train_scores_data[cols].corr(), ax=ax)
img = nib.load(fmri_mask)
print(img.shape)

print(img.get_data_dtype())

print(img.get_data_dtype() == np.dtype(np.float32))

print(img.affine.shape)

data = img.get_fdata()

print(data.shape)

print(type(data))

hdr = img.header

print(hdr.get_xyzt_units())

raw = hdr.structarr

print(raw['xyzt_units'])
df_loading_train = load_data[load_data.Id.isin(train_scores_data.Id)]

df_loading_test = load_data[~load_data.Id.isin(train_scores_data.Id)]
df_loading_train.head()
df_loading_test.head()
# Equally splitted data 

print(df_loading_test.shape)

print(df_loading_train.shape)
df_fnc_train = fnc_data[fnc_data.Id.isin(train_scores_data.Id)]

df_fnc_test = fnc_data[~fnc_data.Id.isin(train_scores_data.Id)]
print(df_fnc_train.shape)

print(df_fnc_test.shape)
df_train_loading_fcn = pd.merge(df_loading_train, df_fnc_train, on = 'Id', how = 'inner')

df_test_loading_fcn =   pd.merge(df_loading_test, df_fnc_test, on = 'Id', how = 'inner')



print(df_train_loading_fcn.shape)

print(df_test_loading_fcn.shape)
columns = df_train_loading_fcn.columns

df_test_loading_fcn = df_test_loading_fcn[columns]
p_value = .00001   # you can change the p-value and experiment

list_p_value =[]



for i in tqdm(columns[1:]):

    list_p_value.append(stats.ks_2samp(df_test_loading_fcn[i] , df_train_loading_fcn[i])[1])



Se = pd.Series(list_p_value, index = columns[1:]).sort_values() 

list_dissimilar = list(Se[Se < p_value].index)
len(list_dissimilar)
df_train_loading_fcn.head()