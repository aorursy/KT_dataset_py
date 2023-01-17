!pip install nodevectors
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score



import itertools

import xgboost 

import lightgbm

import networkx

import nodevectors

import os,gc

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/liverpool-ion-switching/train.csv')

test = pd.read_csv('../input/liverpool-ion-switching/test.csv')

print(train.shape, test.shape)
train.head()
train.info()
test.head()
print(train.isna().any().sum(), test.isna().any().sum())
fig, ax = plt.subplots(1,2,figsize=(20,8))

sns.countplot(train.open_channels, ax=ax[0])

sns.distplot(train.open_channels, ax=ax[1])
# plt.figure(figsize=(20,8))

# sns.scatterplot(x='time', y='open_channels', data=train)
# plt.figure(figsize=(20,8))

# sns.lineplot(x='time', y='signal', hue='open_channels', data=train[train['time'] < 100])
# plt.figure(figsize=(20,8))

# sns.lineplot(x='time', y='signal', hue='open_channels', data=train[(train['time'] > 100) & (train['time'] < 200)])
# plt.figure(figsize=(20,8))

# sns.lineplot(x='time', y='signal', hue='open_channels', data=train[(train['time'] > 200) & (train['time'] < 300)])
# plt.figure(figsize=(20,8))

# sns.lineplot(x='time', y='signal', hue='open_channels', data=train[(train['time'] > 300) & (train['time'] < 400)])
# plt.figure(figsize=(20,8))

# sns.lineplot(x='time', y='signal', hue='open_channels', data=train[train['time'] > 400])
train.describe()
G = networkx.Graph()

G
temp = train[['signal', 'open_channels']]

temp
temp['signal'] = temp['signal'].round(2)

temp['signal'].nunique()
temp['signal'] = temp['signal'].astype(str)

temp
G.add_nodes_from(temp['signal'].values, signal=True)

print(len(G.nodes()))
temp['open_channels'] = temp['open_channels'].apply(lambda x : -99 if x ==0 else x * -999)

temp
G.add_nodes_from(temp['open_channels'].values, channel=True)

print(len(G.nodes()))
G.add_edges_from(temp.values)

print(len(G.nodes()), len(G.edges()))
def createEdgesBetweenSignalsWithSameOpenChannel(G):

    channel_nodes = networkx.get_node_attributes(G, 'channel')

    channel_nodes = list(channel_nodes.keys())

    edges_to_be_created = []



    for i in channel_nodes:

        if(len([x for x in G.neighbors(i)]) > 1):

            edges_to_be_created.append([x for x in G.neighbors(i)])

            

    for i in edges_to_be_created:

        for j in itertools.combinations(i,2):

            G.add_edge(*j)

            

    return G 
G = createEdgesBetweenSignalsWithSameOpenChannel(G)

print(len(G.edges()))
channel_nodes = networkx.get_node_attributes(G, 'channel')

channel_nodes = list(channel_nodes.keys())

G.remove_nodes_from(channel_nodes)
n2v = nodevectors.Node2Vec(

    walklen=32,

    epochs=10,

    return_weight=1,

    neighbor_weight=1.0,

    n_components=32,

    w2vparams={'window': 10,

               'min_count':1

              }

)

n2v.fit(G, verbose=True)
%%time

nodes = [i for i in n2v.model.wv.vocab]

embeddings = np.array([n2v.model.wv[x] for x in nodes])
embedding_df = pd.DataFrame()

embedding_df['signal'] = nodes

embedding_df['embed'] = list(embeddings)

embedding_df
del G, temp

gc.collect()
embed_cols = [f"embed{i}" for i in range(32)]

embed_cols



df3 = pd.DataFrame(embedding_df['embed'].values.tolist(), columns=embed_cols)

df3
embedding_df = embedding_df.join(df3)

embedding_df = embedding_df.drop(['embed'], axis=1)

embedding_df
del df3, embed_cols

gc.collect()
embedding_df = embedding_df.drop_duplicates(subset=['signal'])

embedding_df
embedding_df['signal'] = embedding_df['signal'].astype(float)

embedding_df
train['train'] = 1

test['train'] = 0
all_data = pd.concat([train,test]).drop(['open_channels'], axis=1).reset_index(drop=True)

all_data
all_data = all_data.sort_values(by=['time']).reset_index(drop=True)

all_data
all_data.index = ((all_data.time * 10_000) - 1).values

all_data
all_data['batch'] = all_data.index // 50_000

all_data
all_data['signal_batch_min'] = all_data.groupby('batch')['signal'].transform('min')

all_data['signal_batch_max'] = all_data.groupby('batch')['signal'].transform('max')

all_data['signal_batch_std'] = all_data.groupby('batch')['signal'].transform('std')

all_data['signal_batch_mean'] = all_data.groupby('batch')['signal'].transform('mean')

all_data['signal_batch_median'] = all_data.groupby('batch')['signal'].transform('median')
all_data['signal_batch_skew'] = all_data.groupby('batch')['signal'].transform('skew')

all_data['mean_abs_chg_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.mean(np.abs(np.diff(x))))

all_data['median_abs_chg_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.median(np.abs(np.diff(x))))

all_data['abs_max_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.max(np.abs(x)))

all_data['abs_min_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.min(np.abs(x)))

all_data['abs_mean_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.mean(np.abs(x)))

all_data['abs_median_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.median(np.abs(x)))

all_data['moving_average_batch_1000_mean'] = all_data.groupby(['batch'])['signal'].rolling(window=1000).mean().mean(skipna=True)
all_data['signal_round'] = all_data['signal'].round(2)

all_data
all_data = all_data.merge(embedding_df, left_on='signal_round', right_on='signal', how='left')

all_data
all_data = all_data.drop_duplicates(subset=['time'])

all_data
del test, embedding_df

gc.collect()
KFOLDS = 5

cv = KFold(n_splits=KFOLDS, shuffle=True, random_state=108)
# xgb = xgboost.XGBClassifier(tree_method='hist', objective='multi:softmax')

param_grid ={

    'learning_rate': [0.01],

    'n_estimators':[100],

    }
lgb = lightgbm.LGBMClassifier(objective='multiclass')
clf = GridSearchCV(

        estimator=lgb,

        param_grid=param_grid,

        cv=cv,

        iid=True,

        return_train_score=True,

        scoring='f1_macro',

        verbose=0

    )
clf.fit(all_data[all_data['train']==1].drop(['train', 'time', 'signal_y'],axis=1),train['open_channels'])
plt.figure(figsize=(20,6))

sns.barplot(x=clf.best_estimator_.feature_importances_, y=all_data[all_data['train']==1].drop(['train', 'time', 'signal_y'],axis=1).columns)
X_test = all_data[all_data['train'] == 0]

X_test
predictions = clf.predict(X_test.drop(['train', 'time', 'signal_y'], axis=1))

predictions
plt.figure(figsize=(8,6))

sns.countplot(predictions)
X_test['open_channels'] = predictions

X_test
X_test = X_test[['time', 'open_channels']]

X_test.to_csv('submission.csv', index=False, float_format='%.4f')