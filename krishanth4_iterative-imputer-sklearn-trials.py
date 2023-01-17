# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.experimental import enable_iterative_imputer  

from sklearn.impute import IterativeImputer

from sklearn.metrics import mean_squared_error

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../../kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv')
data.head()
data.drop(columns='profile_id', inplace=True)
data.boxplot(figsize=(12,8), grid=False)
data.describe()
df = data.copy()
def defile_dataset(df, col_selection_rate=0.40):

    cols = np.random.choice(df.columns, int(len(df.columns)*col_selection_rate))

    df_cp = df.copy()

    for col in cols:

        data_drop_rate = np.random.choice(np.arange(0.15, 0.5, 0.02), 1)[0]

        drop_ind = np.random.choice(np.arange(len(df_cp[col])), size=int(len(df_cp[col])*data_drop_rate), replace=False)

        df_cp[col].iloc[drop_ind] = np.nan

    return df_cp, cols
df_mod, cols = defile_dataset(df)
df_mod.info()
imputer = IterativeImputer(sample_posterior=True)

imputer_non = IterativeImputer()
df1 = df_mod.copy()

df2 = df_mod.copy()
df_mod_fit = imputer.fit_transform(df1)
imputer.get_params()
df_mod_fit_non = imputer_non.fit_transform(df2)
df_mod_fit.shape
df_fit_np = df_mod_fit[:,[df.columns.get_loc(i) for i in cols]]
df_fit_non_np = df_mod_fit_non[:,[df.columns.get_loc(i) for i in cols]]
pd.DataFrame(df_fit_non_np, columns=cols)
df_np = df[cols].values
print(df_fit_np.shape)

print(df_np.shape)
for i in range(len(cols)):

    print("When sample posterior is True {:.3f} and when it isnt {:.3f}".format(mean_squared_error(df_np[:,i], df_fit_np[:,i]), mean_squared_error(df_np[:,i], df_fit_non_np[:,i])))
print(np.std(df_np, axis=0))

print(np.std(df_fit_non_np, axis=0))
print(stats.sem(df_np, axis=0))

print(stats.sem(df_fit_non_np, axis=0))
def impute(df_orig):

    df_miss, cols = defile_dataset(df_orig)

    df_orig_slice = df_orig[cols]

    imputed_data = []

    n_iterations = []

    for i in range(10):

        imputer = IterativeImputer(max_iter=(i+1)*10)

        df_stg = df_miss.copy()

        imp_arr = imputer.fit_transform(df_stg)

        n_iterations.append(imputer.n_iter_)

        imp_arr_sl = imp_arr[:,[df_orig.columns.get_loc(i) for i in cols]]

        imputed_data.append(pd.DataFrame(imp_arr_sl, columns=cols))

    return df_orig_slice, imputed_data, n_iterations
def impute_once(df_orig):

    df_miss, cols = defile_dataset(df_orig)

    df_orig_slice = df_orig[cols]

    imputer = IterativeImputer(max_iter=100)

    df_stg = df_miss.copy()

    imp_arr = imputer.fit_transform(df_stg)

    return df_orig_slice, df_miss[cols], pd.DataFrame(imp_arr[:,[df_orig.columns.get_loc(i) for i in cols]], columns=cols), imputer.n_iter_
df_orig, imp_data, n_iter = impute(df)
df_orig.head()
len(imp_data)
cols = df_orig.columns
imp_data[0][cols[0]]
n_iter
fig, axes = plt.subplots(len(cols), sharex=True, figsize=(6,6), dpi=120)

for i in range(len(cols)):

    bars = []

    for j in range(len(imp_data)):

        bars.append(mean_squared_error(df_orig[cols[i]], imp_data[j][cols[i]])) 

    axes[i].bar(np.arange(10), bars, color='white', width=0.6, edgecolor='k', linewidth=1)

    axes[i].set_ylim([0,0.2])

    axes[i].set_yticks([])

    for k in range(len(imp_data)):

        axes[i].text(k-0.2, bars[k]+0.01, round(bars[k], 3), fontsize=6)

    axes[i].set_title(str(cols[i]))

fig.tight_layout()

plt.show()
'''

fig, axes = plt.subplots(len(cols), sharex=True, figsize=(10,8), dpi=120)

for i in range(len(cols)):

    for j in range(len(imp_data)):

        sns.boxplot(np.arange(10), imp_data[j][cols[i]], ax=axes[i])

    #axes[i].set_ylim([0.8,1.2])

    #axes[i].set_yticks([])

    #for k in range(len(imp_data)):

        #axes[i].text(k-0.2, bars[k]+0.01, round(bars[k], 3), fontsize=6)

    #axes[i].set_title(str(cols[i]))

fig.tight_layout()

plt.show()

'''
pd.DataFrame(imp_data[0].stack()).reset_index()
df_og, df_def, df_imp, n_iter = impute_once(df)
print(df_og.columns)

print(df_imp.columns)

print(n_iter)
for i in range(len(df_og.columns)):

    print("Iterative Imputer: MSE for {} is {:.4f}.".format(df_og.columns[i], mean_squared_error(df_og[df_og.columns[i]], df_imp[df_imp.columns[i]])))
df_def = df_def[df_og.columns]
from sklearn.impute import SimpleImputer
sim_imp = SimpleImputer()
df_simimp = pd.DataFrame(sim_imp.fit_transform(df_def), columns=df_og.columns)
for i in range(len(df_og.columns)):

    print("Simple Imputer: MSE for {} is {:.4f}.".format(df_og.columns[i], mean_squared_error(df_og[df_og.columns[i]], df_simimp[df_simimp.columns[i]])))