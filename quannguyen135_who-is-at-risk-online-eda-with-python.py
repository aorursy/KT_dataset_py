import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gc

import warnings
warnings.filterwarnings('ignore')
# for Kaggle, change to
df = pd.read_csv('../input/cleaned-online-sex-work/cleaned_online_sex_work.csv', index_col=0)
#df = pd.read_csv('../input/cleaned_online_sex_work.csv', index_col=0)
df = df.iloc[: 28831, :]
df = df[~ df.index.duplicated(keep='first')]
df.head()
for column in df.columns:
    print(column)

df.shape
train_df = df[df['Risk'].isnull() == False]
train_df['Risk'] = train_df['Risk'].astype(int)
norisk_df = train_df[train_df['Risk'] == 0]
risk_df = train_df[train_df['Risk'] != 0]

print(train_df.shape)
f, ax = plt.subplots(2, 2, figsize=(20, 10))

sns.countplot(x='Female', hue='Risk', data=train_df, ax=ax[0][0])

sns.distplot(norisk_df['Age'], kde_kws={'label': 'Low Risk'}, ax=ax[0][1])
sns.distplot(risk_df['Age'], kde_kws={'label': 'High Risk'}, ax=ax[0][1])

sns.countplot(x='Location', hue='Risk', data=train_df, ax=ax[1][0])

sns.countplot(x='Verification', hue='Risk', data=train_df, ax=ax[1][1])

plt.show()
orientation_df = train_df[['Heterosexual', 'Homosexual', 'bicurious', 'bisexual']].idxmax(1)
orientation_df = pd.concat([orientation_df, train_df['Risk']], axis=1).rename(columns={0: 'Orientation'})
orientation_df.head()
sns.countplot(x='Orientation', hue='Risk', data=orientation_df)

plt.show()
del orientation_df; gc.collect()
polarity_df = train_df[['Dominant', 'Submisive', 'Switch']].idxmax(1)
polarity_df = pd.concat([polarity_df, train_df['Risk']], axis=1).rename(columns={0: 'Polarity'})
polarity_df.head()
sns.countplot(x='Polarity', hue='Risk', data=polarity_df)

plt.show()
del polarity_df; gc.collect()
looking_df = train_df[['Men', 'Men_and_Women', 'Nobody', 'Nobody_but_maybe', 'Women']].idxmax(1)
looking_df = pd.concat([looking_df, train_df['Risk']], axis=1).rename(columns={0: 'Looking_for'})
looking_df.head()
sns.countplot(x='Looking_for', hue='Risk', data=looking_df)

plt.show()
del looking_df; gc.collect()
f, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.distplot(norisk_df['Points_Rank'], kde_kws={'label': 'Low Risk'}, ax=ax[0])
sns.distplot(risk_df['Points_Rank'], kde_kws={'label': 'High Risk'}, ax=ax[1])

plt.show()
f, ax = plt.subplots(2, 2, figsize=(20, 10))

sns.distplot(norisk_df['Number_of_Comments_in_public_forum'], kde_kws={'label': 'Low Risk'}, ax=ax[0][0])
sns.distplot(risk_df['Number_of_Comments_in_public_forum'], kde_kws={'label': 'High Risk'}, ax=ax[0][0])

sns.distplot(norisk_df['Time_spent_chating_H:M'], kde_kws={'label': 'Low Risk'}, ax=ax[0][1])
sns.distplot(risk_df['Time_spent_chating_H:M'], kde_kws={'label': 'High Risk'}, ax=ax[0][1])

sns.distplot(norisk_df['Number_of_advertisments_posted'], kde_kws={'label': 'Low Risk'}, ax=ax[1][0])
sns.distplot(risk_df['Number_of_advertisments_posted'], kde_kws={'label': 'High Risk'}, ax=ax[1][0])

sns.distplot(norisk_df['Number_of_offline_meetings_attended'], kde_kws={'label': 'Low Risk'}, ax=ax[1][1])
sns.distplot(risk_df['Number_of_offline_meetings_attended'], kde_kws={'label': 'High Risk'}, ax=ax[1][1])

plt.show()
f, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.distplot(norisk_df['Number of Friends'], kde_kws={'label': 'Low Risk'}, ax=ax[0])
sns.distplot(risk_df['Number of Friends'], kde_kws={'label': 'High Risk'}, ax=ax[1])

plt.show()
import networkx as nx
import matplotlib.patches as mpatches
network_df = train_df[train_df['Friends_ID_list'].isnull() == False]['Friends_ID_list']
network_df.head()
graph = nx.Graph()
graph.add_nodes_from(list(network_df.index))
for train_id in network_df.index:
    friend_ids = list(map(int, network_df.loc[train_id].split(',')))
    for friend_id in friend_ids:
        graph.add_edge(train_id, friend_id)
f, ax = plt.subplots(1, 1, figsize=(20, 10))

pos = nx.spring_layout(graph)

nodes = nx.draw_networkx_nodes(
    graph,
    pos,
    node_color = 'y',
    node_size = 50
)
nodes.set_edgecolor('black')

norisk_nodelist = list(norisk_df[norisk_df['Friends_ID_list'].isnull() == False].index)
risk_nodelist = list(risk_df[risk_df['Friends_ID_list'].isnull() == False].index)
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist = norisk_nodelist,
    node_color = 'b',
    node_size = 50
)
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist = risk_nodelist,
    node_color = 'r',
    node_size = 50
)

labels = {}
for node in norisk_nodelist: labels[node] = node
for node in risk_nodelist: labels[node] = node
nx.draw_networkx_labels(graph, pos, labels, font_size=10)

nx.draw_networkx_edges(graph, pos, edge_color='y')

patches = [
    mpatches.Patch(color='y', label='Risk Undetermined'),
    mpatches.Patch(color='b', label='Low Risk'),
    mpatches.Patch(color='r', label='High Risk')
]
plt.legend(handles=patches)

plt.show()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

X_train = train_df.drop(['Friends_ID_list', 'Risk'], axis=1)
y_train = train_df['Risk']
X_train.head()
y_train.head()
try:
    clf = LinearSVC()
    clf.fit(X_train, y_train)
except Exception as e:
    print('There was a problem: %s' % str(e))
location_means = train_df.groupby('Location').mean()['Risk']
location_means
train_df['Location'] = train_df['Location'].map(location_means)
train_df['Location'].head()
X_train = train_df.drop(['Friends_ID_list', 'Risk'], axis=1)
try:
    clf = LinearSVC()
    clf.fit(X_train, y_train)
except Exception as e:
    print('There was a problem: %s' % str(e))
nfeatures = 10

coef = clf.coef_.ravel()
top_positive_coefs = np.argsort(coef)[-nfeatures :]
top_negative_coefs = np.argsort(coef)[: nfeatures]
top_coefs = np.hstack([top_negative_coefs, top_positive_coefs])

plt.figure(figsize=(15, 5))
colors = ['red' if c < 0 else 'blue' for c in coef[top_coefs]]
plt.bar(np.arange(2 * nfeatures), coef[top_coefs], color = colors)
feature_names = np.array(X_train.columns)
plt.xticks(np.arange(0, 1 + 2 * nfeatures), feature_names[top_coefs], rotation=60, ha='right')

plt.show()
corr_matrix = train_df.drop(['Friends_ID_list'], axis=1).corr()

f, ax = plt.subplots(1, 1, figsize=(15, 10))
sns.heatmap(corr_matrix)

plt.show()