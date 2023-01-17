import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_y = pd.read_csv('../input/government-bonds/yields.csv').sort_values(by='time', ascending=True)
df_y.tail()
df_y.info(null_counts=True)
syms = df_y.columns.tolist()[1:]

terms = {}

for sym in syms:
    prefix = sym[0:2]
    term = sym[2:]

    if prefix in terms:
        terms[prefix].append(term)
    else:
        terms[prefix] = [term]

terms
sns.lineplot(x='time', y='DE10', data=df_y[['time', 'DE10']]);
df_y_france = df_y[filter(lambda name: name == 'time' or name.startswith('FR'), df_y.columns.tolist())]
df_y_france.tail()
sns.lineplot(x='time', y='FR05', data=df_y_france[['time', 'FR05']]);
sns.lineplot(x='time', y='FR10', data=df_y_france[['time', 'FR10']]);
sns.lineplot(x='time', y='FR20', data=df_y_france[['time', 'FR20']]);
df_p = pd.read_csv('../input/government-bonds/prices.csv').sort_values(by='time', ascending=True)
df_p.tail()
df_p.info(null_counts=True)
df_p_uk = df_p[filter(lambda name: name == 'time' or name.startswith('GB'), df_p.columns.tolist())]
df_p_uk.tail()
sns.lineplot(x='time', y='GB10', data=df_p_uk[['time', 'GB10']]);