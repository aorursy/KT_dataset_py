import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb







d = pd.read_csv('../input/expenditures.csv')



d['CleanY'] = d['Amount'].replace('[\$,]', '', regex=True).astype(float)





#%%



plt.rcParams['figure.figsize']=(8,16)

purp_plot = sb.barplot('CleanY','Payee state', data=d)

plt.xticks(rotation=90)
plt.rcParams['figure.figsize']=(8,6)

purp_plot = sb.barplot('CleanY','Entity type', data=d)

plt.xticks(rotation=90)
plt.rcParams['figure.figsize']=(8,51)

purp_plot = sb.barplot('CleanY','Payee name', data=d)

plt.xticks(rotation=90)
plt.rcParams['figure.figsize']=(8,36)

purp_plot = sb.barplot('CleanY','Purpose', data=d)

plt.xticks(rotation=90)