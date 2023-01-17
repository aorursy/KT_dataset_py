import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.chdir("/kaggle/input/habermans-survival-data-set/")
df = pd.read_csv("haberman.csv")

df.columns=['age', 'op_year', 'axil_nodes', 'survived']
import numpy as np

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df.head()
df['survived'] = df['survived'].map({1:"yes", 2:"no"})

df['survived'] = df['survived'].astype('category')
profile = df.profile_report(title='Pandas Profiling before Data Preprocessing', style={'full_width':True})

#profile.to_file(output_file="profiling_before_preprocessing.html")
profile
df.info()
df.describe()
sns.set_style('dark')

sns.pairplot(df, hue='survived', height=4, diag_kind='kde', palette="cubehelix")
sns.set(style="darkgrid")

ax = sns.countplot(x="op_year", hue="survived", data=df)
for (idx, feature) in enumerate(df.columns[:-1]):

    sns.FacetGrid(df, hue="survived", height=5, palette = "cubehelix").map(sns.distplot, feature).add_legend();

    plt.show();
fig, axarr = plt.subplots(1, 3, figsize = (20, 5))

for (idx, feature) in enumerate(df.columns[:-1]):

    sns.boxplot(

        x = 'survived',

        y = feature,

        palette = "cubehelix",

        data = df,

        ax = axarr[idx]

    )    

plt.show()