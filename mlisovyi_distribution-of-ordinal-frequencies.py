import os

import numpy as np

import pandas as pd

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



DATA = '/kaggle/input/cat-in-the-dat/'



print(os.listdir(DATA))

df_trn = pd.read_csv(f'{DATA}/train.csv').drop('id',axis=1)
# Define category order

cat_order = {'ord_1': (['Novice', 'Contributor','Expert','Master','Grandmaster'],'-'),

             'ord_2': (['Freezing','Cold','Warm','Hot','Boiling Hot', 'Lava Hot'],':'),

             'ord_3': (None,'-.'),

             'ord_4': (None,'--'), 

             'ord_0': ([1,2,3],'-'),

             'ord_5': (None,':')

            }



plt.figure(figsize=(16,8), )

for k,(v,ls) in cat_order.items():

    print(f'{k}: set size = {df_trn[k].nunique()}')

    # value counts for the categorical feature

    x = df_trn[k].value_counts()

    # define custom category order if needed

    if v is not None:

        x.index = pd.CategoricalIndex(data=x.index, categories=v, ordered=True)

    # sort by category

    x = x.sort_index()

    # show only first 30 entries, as ord_5 has way roo many categories

    x = x.head(30)

    # plot normalised frequency distributions

    (x/x.iloc[1]).plot(alpha=0.5, ls=ls, lw=2)

# show legend

plt.legend()

# save into file

plt.savefig('cat_distribution.png')