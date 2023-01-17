# Load packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

sns.set_style('whitegrid')

pd.set_option('display.max_columns', 500)
# Read data. Bin variable values into 50 and 100 bins.

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv') 



coltouse = [col for col in train.columns.tolist() if col not in ['ID_code', 'target']]



for col in tqdm(coltouse):

    out = pd.qcut(train[col], q=50, labels=False, precision=5)

    train[f'{col}_bin50'] = out.values

    

for col in tqdm(coltouse):

    out = pd.qcut(train[col], q=100, labels=False, precision=5)

    train[f'{col}_bin100'] = out.values    
# Draw charts. Each chart represents dinamics of average target per bin.

# Each bin consists of almost equal number of observations.

# Green dots - bin values (i.e. average target for that bin)

# Red line - average target across all dataset.

# Purple lines - conditional borders equals to average target +/- 0.01.

# Each variable is plotted on two scales - 50 and 100 bins. 

# Plots are hidden. If you'd like to look at them - press "Output" button.

for col in coltouse:

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

    plt.tight_layout()

    

    ax1.set_title(f"Distribution of {col}_bin50. Nunique is {train[col].nunique()}")

    ax1.plot(train.groupby([f'{col}_bin50'])['target'].agg('mean'), 'b:')

    ax1.plot(train.groupby([f'{col}_bin50'])['target'].agg('mean'), 'go')

    ax1.axhline(y=train['target'].mean(), color='r', linestyle='-')

    ax1.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')

    ax1.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')

    

    ax2.set_title(f"Distribution of {col}_bin100. Nunique is {train[col].nunique()}")

    ax2.plot(train.groupby([f'{col}_bin100'])['target'].agg('mean'), 'b:')

    ax2.plot(train.groupby([f'{col}_bin100'])['target'].agg('mean'), 'go')

    ax2.axhline(y=train['target'].mean(), color='r', linestyle='-')

    ax2.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')

    ax2.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')

    

    plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

plt.tight_layout()



ax1.set_title(f"Distribution of var_34_bin50. Nunique is {train[col].nunique()}")

ax1.plot(train.groupby([f'var_34_bin50'])['target'].agg('mean'), 'b:')

ax1.plot(train.groupby([f'var_34_bin50'])['target'].agg('mean'), 'go')

ax1.axhline(y=train['target'].mean(), color='r', linestyle='-')

ax1.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')

ax1.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')



ax2.set_title(f"Distribution of var_40_bin100. Nunique is {train[col].nunique()}")

ax2.plot(train.groupby([f'var_40_bin100'])['target'].agg('mean'), 'b:')

ax2.plot(train.groupby([f'var_40_bin100'])['target'].agg('mean'), 'go')

ax2.axhline(y=train['target'].mean(), color='r', linestyle='-')

ax2.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')

ax2.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')



plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

plt.tight_layout()



ax1.set_title(f"Distribution of var_29_bin50. Nunique is {train[col].nunique()}")

ax1.plot(train.groupby([f'var_29_bin50'])['target'].agg('mean'), 'b:')

ax1.plot(train.groupby([f'var_29_bin50'])['target'].agg('mean'), 'go')

ax1.axhline(y=train['target'].mean(), color='r', linestyle='-')

ax1.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')

ax1.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')



ax2.set_title(f"Distribution of var_38_bin100. Nunique is {train[col].nunique()}")

ax2.plot(train.groupby([f'var_38_bin100'])['target'].agg('mean'), 'b:')

ax2.plot(train.groupby([f'var_38_bin100'])['target'].agg('mean'), 'go')

ax2.axhline(y=train['target'].mean(), color='r', linestyle='-')

ax2.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')

ax2.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')



plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

plt.tight_layout()



ax1.set_title(f"Distribution of var_80_bin50. Nunique is {train[col].nunique()}")

ax1.plot(train.groupby([f'var_80_bin50'])['target'].agg('mean'), 'b:')

ax1.plot(train.groupby([f'var_80_bin50'])['target'].agg('mean'), 'go')

ax1.axhline(y=train['target'].mean(), color='r', linestyle='-')

ax1.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')

ax1.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')



ax2.set_title(f"Distribution of var_108_bin100. Nunique is {train[col].nunique()}")

ax2.plot(train.groupby([f'var_108_bin100'])['target'].agg('mean'), 'b:')

ax2.plot(train.groupby([f'var_108_bin100'])['target'].agg('mean'), 'go')

ax2.axhline(y=train['target'].mean(), color='r', linestyle='-')

ax2.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')

ax2.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')



plt.show()