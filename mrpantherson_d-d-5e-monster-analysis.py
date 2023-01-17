# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/dnd-5e-monsters/dnd_monsters.csv')

df.info()
results = df.loc[:, 'size'].value_counts()

sns.barplot(results.values, results.index, orient='h', color='#84a9ac')
results = df.loc[:, 'type'].value_counts()

len(results)
results.sample(n=20)
df['type'] = df['type'].apply(lambda x: x.split(' (')[0])

results = df.loc[:, 'type'].value_counts()

sns.barplot(results.values, results.index, orient='h', color='#84a9ac')
df.loc[df['type']=='ooze', :]
sns.heatmap(df.corr(), annot=True)
df['legendary'] = df['legendary'].fillna('Normal')

sns.scatterplot(df['hp'], df['ac'], hue=df['legendary'])
results = pd.pivot_table(df, values='cr', index='align', columns='type', aggfunc='count', fill_value=0)

sns.heatmap(results)
fig, ax = plt.subplots(1,2, figsize=(16,8))



sns.boxplot('hp', 'type', data=df, ax=ax[0], color='#84a9ac')

sns.boxplot('hp', 'size', data=df, ax=ax[1], color='#84a9ac')



ax[0].axvline(x=df.hp.mean(), ymin=0, ymax=1, linestyle=':', color='black')

ax[1].axvline(x=df.hp.mean(), ymin=0, ymax=1, linestyle=':', color='black')



ax[0].text(x=df.hp.mean()+5, y=14, s='HP Mean')

ax[1].text(x=df.hp.mean()+5, y=5, s='HP Mean')



ax[0].set_title('Type')

ax[1].set_title('Size')