# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%pylab inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))









# Any results you write to the current directory are saved as output.
limit_rows = 7000000

df = pd.read_csv('../input/train_ver2.csv', dtype = {"sexo": str, "ind_nuevo": str, "ult_fec_cli_1t":str, "index":str}, nrows=limit_rows)
df.describe() # this gives an overall stats of each of the variables
unique_ids = pd.Series(df["ncodpers"].unique())

limit = 10000

unique_ids = unique_ids.sample(n=limit)

df = df[df['ncodpers'].isin(unique_ids)]
len(df)
df.describe()
df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], format='%Y-%m-%d')

df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], format='%Y-%m-%d')

df['fecha_dato'].unique()
df['month'] = pd.DatetimeIndex(df['fecha_dato']).month

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df.isnull().any()
with sns.plotting_context("notebook", font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(df['age'].dropna(), bins=1800, kde=False)

    sns.plt.title("Age Distribution")

    sns.plt.ylabel("Count")

    
df.loc[df['age'] < 18] = df.loc[(df['age'] > 18) & (df['age'] < 30), 'age'].mean(skipna=True)

df.loc[df['age'] > 100] = df.loc[(df['age'] > 30) & (df['age'] < 100), 'age'].mean(skipna=True)

with sns.plotting_context("notebook", font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(df['age'].dropna(), bins=80, kde=False)

    sns.plt.title("Age Distribution")

    sns.plt.ylabel("Count")
# check the missing value for 

df['ind_nuevo'].isnull().sum()
months_active = df.loc[df['ind_nuevo'].isnull(), :].groupby('ncodpers').size()

months_active.max()
# So these are new customers add 1

df.loc[df['ind_nuevo'].isnull(), 'ind_nuevo'] = 1
df['ind_nuevo'].isnull().any()
df.antiguedad = pd.to_numeric(df.antiguedad, errors='coerce')
df.antiguedad.isnull().sum()
# Same number as above may be the same people we have found out before

df.loc[df.antiguedad.isnull(), 'ind_nuevo'].describe()

df.loc[df.antiguedad.isnull(), 'antiguedad'] = df.antiguedad.min()

df.loc[df.antiguedad < 0, 'antiguedad'] = 0
#