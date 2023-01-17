import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

%config InlineBackend.figure_format = 'svg'
df = pd.read_csv('../input/telecom_churn.csv')
df.head()
df.info();
features = ['total day minutes', 'total intl calls']
df[features].hist(figsize=(10,4));
df[features].plot(kind='density', subplots=True, layout=(1,2), 
                  sharex=False, figsize=(10,4));
x = df['total intl calls']
sns.distplot(x);
sns.boxplot(x = 'total intl calls', data = df);
_, axes = plt.subplots(1,2, sharey=True, figsize=(6,4))
sns.boxplot(data=df['total intl calls'], ax=axes[0]);
sns.violinplot(data=df['total intl calls'], ax = axes[1]);
df[features].describe()
df['churn'].value_counts()
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,2))

sns.countplot(x='churn', data = df, ax = axes[0])
sns.countplot(x='customer service calls', data = df, ax = axes[1]);
numerical = list(set(df.columns)-set(['state', 'international plan', 'voice mail plan',
                                     'area code', 'churn', 'customer service calls']))
corr_matrix = df[numerical].corr()
corr_matrix
sns.heatmap(corr_matrix);
numerical = list(set(numerical)-set(['total day charge', 'total eve charge', 'total night charge', 'total intl charge']))
corr_matrix = df[numerical].corr()
sns.heatmap(corr_matrix);
plt.scatter(df['total day minutes'], df['total night minutes']);
sns.jointplot(x='total day minutes', y='total night minutes', data=df, kind = 'scatter');
sns.jointplot('total day minutes', 'total night minutes', data = df, kind = 'kde', color='g');
# pairplot() may become very slow with the SVG format
%config InlineBackend.figure_format = 'png'
sns.pairplot(df[numerical]);
%config InlineBackend.figure_format = 'svg'
sns.lmplot('total day minutes', 'total night minutes', data = df, hue='churn', fit_reg=False);
numerical = list(set(numerical)-set(['phone number']))
# sometimes we can analyse an ordinal variables just as numerical one
numerical.append('customer service calls')

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10,7))
for idx, feat in enumerate(numerical):
    ax = axes[int(idx / 4), idx % 4]
    sns.boxplot(x='churn', y = feat, data=df, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(feat)
fig.tight_layout();
_, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

sns.boxplot(x='churn', y='total day minutes', data = df, ax = axes[0]);
sns.violinplot(x='churn', y='total day minutes', data = df, ax = axes[1]);
sns.catplot(x='churn', y = 'total day minutes', col='customer service calls',
           data=df[df['customer service calls'] < 8], kind = 'box', col_wrap=4, height=3, aspect=0.8);
sns.countplot(x='customer service calls', hue='churn', data=df);
_, axes = plt.subplots(1, 2, sharey = True, figsize=(10,4))

sns.countplot(x = 'international plan', hue = 'churn', data = df, ax = axes[0]);
sns.countplot(x = 'voice mail plan', hue = 'churn', data = df, ax = axes[1]);
pd.crosstab(df['state'], df['churn'])
df.groupby(['state'])['churn'].agg([np.mean]).sort_values(by='mean', ascending=False)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
X = df.drop(['churn', 'state', 'phone number'], axis = 1)
X['international plan'] = X['international plan'].map({'yes':1, 'no':0})
X['voice mail plan'] = X['voice mail plan'].map({'yes':1,'no':0})
X.head()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
%%time
tsne = TSNE(random_state=17)
tsne_repr = tsne.fit_transform(X_scaled)
plt.scatter(tsne_repr[:,0], tsne_repr[:,1], alpha=0.5);
plt.scatter(tsne_repr[:,0], tsne_repr[:,1], c=df['churn'].map({False:'blue', True:'orange'}), alpha=0.5);
_, axes = plt.subplots(1, 2, sharey=True, figsize=(12,5))

for i, name in enumerate(['international plan', 'voice mail plan']):
    axes[i].scatter(tsne_repr[:,0], tsne_repr[:,1],
                c=df[name].map({'yes' : 'orange', 'no':'blue'}), alpha=0.5);
    axes[i].set_title(name);