import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from pylab import scatter
# Colour palette for plotting

clr_pal = ['#000000','#e6194b','#3cb44b','#ffe119','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe','#008080','#e6beff','#aa6e28','#fffac8','#800000','#808000','#ffd8b1','#000080','#808080','#aaffc3']
data = pd.read_csv('../input/Pokemon.csv')

features = data.columns

data.head(10)
data.isnull().any(axis=0)
data['Type 2'] = data['Type 2'].fillna("Basic")

data.head(10)
df1 = data.drop(['#','Total'], axis=1)

plt.subplots(figsize = (15,10))

sns.boxplot(data=df1)

sns.plt.show()
plt.subplots(1,2,figsize=(15,10))

plt.subplot(2,2,1)

sns.distplot(data['Generation'],bins=100)

plt.subplot(2,2,2)

sns.distplot(data['Legendary'],bins=100)

sns.plt.show()
df2 = data.groupby(['Type 1'], as_index=True).mean()

df2 = df2.drop(['#','Total','Legendary','Generation'],axis=1).transpose()

df2
df2[df2.columns].plot(color=clr_pal[:len(df2.columns)],marker='D')

plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

plt.gcf().set_size_inches(15,10)

plt.show()
plt.subplots(figsize=(15,10))

sns.heatmap(df2.corr(), annot=True)

sns.plt.show()
df3 = data.groupby(['Type 1'], as_index=True).mean()

df3 = df3.drop('#',axis=1)

plt.subplots(figsize=(15,10))

sns.heatmap(df3.corr(), annot=True)

sns.plt.show()
def feature_relation(feature1, feature2):

    plot_features = ['Type 1','population',feature1, feature2]

    df4 = pd.DataFrame(columns=plot_features)

    df4[plot_features[0]] = data.groupby(['Type 1']).mean().index.tolist()

    df4[plot_features[1]] = data.groupby(['Type 1'])['#'].count().tolist()

    df4[plot_features[2]] = data.groupby(['Type 1'])[plot_features[2]].mean().tolist()

    df4[plot_features[3]] = data.groupby(['Type 1'])[plot_features[3]].mean().tolist()

    

    fig, ax = plt.subplots()

    fig.set_size_inches(15,10)

    for idx,val in df4.iterrows():

        ax.scatter(x= val[feature1], y=val[feature2], c = clr_pal[idx], label=val['Type 1'], s=val['population']*20)

    lgnd = plt.legend(df4['Type 1'],  bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    ax.set_xlabel(feature1)

    ax.set_ylabel(feature2)

    for x in range(len(df4.index)):

        lgnd.legendHandles[x]._sizes = [100]

    plt.show()
feature_relation('Sp. Atk','Sp. Def')
feature_relation('Speed','Defense')