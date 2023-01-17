import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



pd.set_option('max_columns', None)



train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

synthetic = np.load('../input/list-of-fake-samples-and-public-private-lb-split/synthetic_samples_indexes.npy')

train = train.sample(frac=0.5)

test_real = test.iloc[~test.index.isin(synthetic)]

test_fake = test.iloc[test.index.isin(synthetic)]

data = pd.concat([train, test_real, test_fake], axis=0, sort=False)



print(len(train))

print(len(test_real))

print(len(test_fake))
# === unique value

col_var = train.columns[2:]

df = pd.DataFrame(col_var, columns=['feature'])

df['n_train_unique'] = train[col_var].nunique(axis=0).values

df['n_test_real_unique'] = test_real[col_var].nunique(axis=0).values

df['n_test_fake_unique'] = test_fake[col_var].nunique(axis=0).values



for i in df.index:

    col = df.loc[i, 'feature']

    df.loc[i, 'n_overlap_withreal'] = int(np.isin(train[col].unique(), test_real[col]).sum())



df['value_range'] = data[col_var].max(axis=0).values - data[col_var].min(axis=0).values
df.T
# === plot

df = df.sort_values(by='n_train_unique').reset_index(drop=True)

df[['n_train_unique', 'n_test_real_unique', 'n_test_fake_unique', 'n_overlap_withreal']].plot(kind='barh' ,figsize=(22, 100), fontsize=20, width=0.8)

plt.yticks(df.index, df['feature'].values)

plt.xlabel('n_unique', fontsize=20)

plt.ylabel('feature', fontsize=20)

plt.legend(loc='center right', fontsize=20)