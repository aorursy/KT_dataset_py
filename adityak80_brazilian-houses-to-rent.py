import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/brasilianhousestorentclean/clean_data.csv', index_col=0)

df = df.iloc[:,:-3]

df
# fig,ax=plt.subplots(figsize=(17,1))

# sns.heatmap(corr.sort_values(by='total', ascending=False).head(1), cmap='Reds');

# corr.sort_values(by='total')['total']
sns.distplot(df['rent amount']);
cols = ['area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'hoa', 'total', 'property tax', 'fire insurance']

fig,ax = plt.subplots(nrows=3, ncols=3, figsize=(12,12))

for i,x in enumerate(cols):

    sns.distplot(df[x], ax=ax[i//3][i%3])

plt.tight_layout()
bin_cols = ['city','animal', 'furniture']

fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

for i, c in enumerate(bin_cols):

    df[c].value_counts().plot(kind='bar', ax=ax[i])

    ax[i].set_xlabel(c)

    ax[i].set_ylabel('Counts')

plt.tight_layout()
bin_cols = ['city','animal', 'furniture']

fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

for i,x in enumerate(bin_cols):

    sns.boxplot(x=x, y='rent amount',data=df, ax=ax[i%3])

    ax[i%3].set_ylim(0,50000);



plt.tight_layout()
cols = ['area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'hoa', 'total', 'property tax', 'fire insurance']

fig,ax = plt.subplots(nrows=3, ncols=3, figsize=(12,12))

for i,x in enumerate(cols):

    sns.scatterplot(x=x,y='rent amount',data=df, ax=ax[i//3][i%3]);

plt.tight_layout()
def plot_correlations(frame, features):

    spr = pd.DataFrame()

    spr['feature'] = features

    spr['spearman'] = [frame[f].corr(frame['rent amount'], 'spearman') for f in features]

    spr['pearson'] = [frame[f].corr(frame['rent amount'], 'pearson') for f in features]

    spr = spr.sort_values('spearman')

    fig , ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 0.25*len(features)))

    sns.barplot(data=spr, y='feature', x='spearman', orient='h', ax=ax[0])

    spr = spr.sort_values('pearson')

    sns.barplot(data=spr, y='feature', x='pearson', orient='h', ax=ax[1])

    plt.tight_layout()

    

# features = quantitative + qual_encoded

plot_correlations(df, features=df.columns)