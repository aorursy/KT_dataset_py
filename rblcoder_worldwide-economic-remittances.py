import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
df_remittance_outflow = pd.read_csv('/kaggle/input/worldwide-economic-remittances/remittance-outflow.csv')
df_remittance_outflow.tail(10)
df_remittance_outflow = df_remittance_outflow[:-8]
df_remittance_outflow.columns
df_remittance_outflow.set_index('Migrant remittance outflows (US$ million)').drop('Unnamed: 0', axis=1).T.iloc[:].columns
df_remittance_outflow.set_index('Migrant remittance outflows (US$ million)').drop('Unnamed: 0', axis=1).T.iloc[:]
df_remittance_inflow = pd.read_csv('/kaggle/input/worldwide-economic-remittances/remittance-inflow.csv')

df_remittance_inflow.tail(10)
df_remittance_inflow.set_index('Migrant remittance inflows (US$ million)').drop('Unnamed: 0', axis=1).T.iloc[:, :-8]
print(len(data_i.columns))

print(len(data_o.columns))


f, axes = plt.subplots(nrows=214, ncols=2, figsize=(19, 900), sharex=True, sharey=True)

data_i = df_remittance_inflow.set_index('Migrant remittance inflows (US$ million)').drop('Unnamed: 0', axis=1).T.iloc[:, :-8]

data_o = df_remittance_outflow.set_index('Migrant remittance outflows (US$ million)').drop('Unnamed: 0', axis=1).T.iloc[:]

cols = data_i.columns

ctr=0

for col in cols:

    #_=plt.scatter(x=data_i.index, y=data_i[col], ax=axes[ctr, 0])

    #_=plt.scatter(x=data_o.index, y=data_o[col], ax=axes[ctr, 1])

    axes[ctr, 0].scatter(x=data_i.index, y=data_i[col])

    axes[ctr, 0].title.set_text(col + ' remittance inflows')

    axes[ctr, 0].xaxis.set_tick_params(rotation=45)

    axes[ctr, 0].set_yscale('log')

    axes[ctr, 1].scatter(x=data_o.index, y=data_o[col])

    axes[ctr, 1].title.set_text(col + ' remittance outflows')

    axes[ctr, 1].xaxis.set_tick_params(rotation=45)

    axes[ctr, 1].set_yscale('log')

   # _=f.suptitle(col)

    ctr+=1