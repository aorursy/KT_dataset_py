import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df_import = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')

df_export = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')
df_import.isnull().sum()
df_export.isnull().sum()
df_import.dropna(inplace=True)

df_export.dropna(inplace=True)
imports_year = df_import.groupby(['year']).sum()['value'].tolist()

exports_year = df_export.groupby(['year']).sum()['value'].tolist()
df_trades_year = pd.DataFrame({'year': list(range(2010, 2019)),'exports_year': exports_year, 'imports_year': imports_year})
df_trades_year['surplus'] = df_trades_year['exports_year'] - df_trades_year['imports_year']  
df_trades_year_melted = df_trades_year.melt(id_vars='year', value_vars=['exports_year', 'imports_year', 'surplus'])
sns.lineplot(x='year', y='value', hue='variable', data=df_trades_year_melted)