%matplotlib inline

import matplotlib.pylab as plt

import pandas as pd

import numpy as np
df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])
def plot_inspect_missings(df, stratifier, vars=None, min_freq=0):

    """Creates 2D colormap of completeness by columns given a stratification column.



         Args:

             df: Pandas dateframe.

             stratifier: Column name used for stratification.

             vars: List of column names to check (all if None)

             min_freq: Only plot the columns that have at least this nan frequency

         """

    if vars is None:

        vars = df.columns.values.tolist()

    if stratifier in vars:

        temp = df.loc[:, vars]

    else:

        temp = df.loc[:, vars + [stratifier]]

    included_vars = []

    for i in vars:

        if i != stratifier:

            if df[i].dtype == int or df[i].dtype == float or df[i].dtype == object:

                if df[i].isnull().sum() > min_freq:

                    included_vars.append(i)

                    temp[i] = df[i].notnull()

            else:

                temp[i] = df[i].fillna('')

                if (df[i] == '').sum() > min_freq:

                    included_vars.append(i)

                    temp[i] = df[i] != ''

    temp = temp.loc[:, included_vars + [stratifier]]

    try:

        temp[stratifier] = temp[stratifier].astype(int)

    except:

        n = 1

    a = temp.groupby(temp[stratifier]).mean().transpose()

    plt.figure(num=None, figsize=(18, 18), dpi=80, facecolor='w', edgecolor='k')

    plt.pcolor(a*100, cmap='RdYlGn', vmin=0, vmax=100)

    plt.yticks(np.arange(0.5, len(a.index), 1), a.index.values)

    plt.xticks(np.arange(0.5, len(a.columns), 1), a.columns.values, rotation='vertical')

    plt.colorbar(label='Completeness')

    plt.ylabel('Data fields')

    plt.xlabel(stratifier)

    plt.title('Completeness')
df_macro['ym'] = df_macro.timestamp.dt.year.astype(str) + '_' + df_macro.timestamp.dt.month.astype(str).str.zfill(2)

plot_inspect_missings(df_macro, 'ym')
df_train['ym'] = df_train.timestamp.dt.year.astype(str) + '_' + df_train.timestamp.dt.month.astype(str).str.zfill(2)

plot_inspect_missings(df_train, 'ym')
plot_inspect_missings(df_train, 'sub_area')
df_test['ym'] = df_test.timestamp.dt.year.astype(str) + '_' + df_test.timestamp.dt.month.astype(str).str.zfill(2)

plot_inspect_missings(df_test, 'ym')
plot_inspect_missings(df_test, 'sub_area')