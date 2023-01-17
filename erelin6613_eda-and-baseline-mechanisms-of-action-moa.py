import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow.keras.layers as L

from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split



%matplotlib inline

plt.rcParams['figure.figsize'] = (12, 5);

sns.set_style('whitegrid')
root_dir = '../input/lish-moa/'

dfs = {}



for file in os.listdir(root_dir):

    dfs[file.split('.')[0]] = pd.read_csv(os.path.join(root_dir, file))

for k, v in dfs.items():

    print(k)

    print(v.shape)

    print('-'*10)
for k in ['test_features', 'train_features']:

    print(k)

    compound_cols = [x for x in dfs[k].keys() if x.startswith('cp_')]

    gene_cols = [x for x in dfs[k].keys() if x.startswith('g-')]

    cell_cols = [x for x in dfs[k].keys() if x.startswith('c-')]

    print('compound features:', len(compound_cols),

         'gene features:', len(gene_cols),

         'cell features:', len(cell_cols))

    """

    uncomment if needed

    print('compounds:')

    print(dfs[k][compound_cols].describe())

    print('genes:')

    print(dfs[k][gene_cols].describe())

    print('cells:')

    print(dfs[k][cell_cols].describe())

    """

    print('~'*10)
for c in compound_cols:

    print('train:\n', dfs['train_features'][c].value_counts())

    print('test:\n', dfs['test_features'][c].value_counts())
fig, ax = plt.subplots(2, 2)

gs = [dfs['train_features'][gene_cols[x]] for x in np.random.choice(list(range(len(gene_cols))), size=4)]

sns.distplot(gs[0], ax=ax[0][0]);

sns.distplot(gs[1], ax=ax[0][1]);

sns.distplot(gs[2], ax=ax[1][0]);

sns.distplot(gs[3], ax=ax[1][1]);

plt.tight_layout();
fig, ax = plt.subplots(2, 2)

gs = [dfs['train_features'][cell_cols[x]] for x in np.random.choice(list(range(len(cell_cols))), size=4)]

sns.distplot(gs[0], ax=ax[0][0], color='gold');

sns.distplot(gs[1], ax=ax[0][1], color='gold');

sns.distplot(gs[2], ax=ax[1][0], color='gold');

sns.distplot(gs[3], ax=ax[1][1], color='gold');

plt.tight_layout();
for df in dfs.values():

    print(df.isna().sum().sum())
dfs['train_targets_scored'].head(3)
dfs['train_targets_scored'].drop('sig_id', axis=1).sum().sort_values()
corr_t = dfs['train_targets_scored'].corr().abs()

sns.heatmap(corr_t[corr_t>0.75], cmap='OrRd');
temp_df = pd.merge(dfs['train_targets_scored'],

                    dfs['train_features'], on='sig_id')

corr = temp_df.corr()
corr.dropna(inplace=True)

sns.heatmap(corr, cmap='coolwarm');
del temp_df
targets = dfs['train_targets_scored'].drop('sig_id', axis=1).sum()

    

least = sorted(targets.items(), key=lambda x: x[1])[:10]

most = sorted(targets.items(), key=lambda x: x[1])[-10:]

least_x = [i[0] for i in least]

least_y = [i[1] for i in least]

most_x = [i[0] for i in most]

most_y = [i[1] for i in most]

fig, ax = plt.subplots(1, 2)

ax[0].bar(least_x, least_y, color='purple');

ax[0].tick_params(axis='x', labelrotation=75);

ax[1].bar(most_x, most_y, color='purple');

ax[1].tick_params(axis='x', labelrotation=75);
targets_ns = dfs['train_targets_nonscored'].drop('sig_id', axis=1).sum()

    

least = sorted(targets_ns.items(), key=lambda x: x[1])[:10]

most = sorted(targets_ns.items(), key=lambda x: x[1])[-10:]

least_x = [i[0] for i in least]

least_y = [i[1] for i in least]

most_x = [i[0] for i in most]

most_y = [i[1] for i in most]

fig, ax = plt.subplots(1, 2)

ax[0].bar(least_x, least_y, color='green');

ax[0].tick_params(axis='x', labelrotation=75);

ax[1].bar(most_x, most_y, color='green');

ax[1].tick_params(axis='x', labelrotation=75);
len(targets_ns[targets_ns==0])
X = dfs['train_features'].drop('sig_id', axis=1)

X['cp_type'] = X['cp_type'].apply(lambda x: 0 if x=='trt_cp' else 1)

X['cp_time'] = X['cp_time'].apply(lambda x: 0 if x==24 else 1)

X['cp_dose'] = X['cp_dose'].apply(lambda x: 0 if x=='D1' else 1)

y = dfs['train_targets_scored'].drop('sig_id', axis=1)



X_train, X_test, y_train, y_test = train_test_split(

    X.values, y.values,

    test_size=0.25)



model = Sequential()

model.add(L.Input(shape=X.shape[1]))

model.add(L.Dense(512, activation='relu'))

model.add(L.Dense(y.shape[1], activation='sigmoid'))



model.compile(loss='binary_crossentropy', 

              optimizer='adam', 

              metrics=['binary_crossentropy', 'accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=5)

model.evaluate(X_test, y_test)
X = dfs['test_features'].drop('sig_id', axis=1)

X['cp_type'] = X['cp_type'].apply(lambda x: 0 if x=='trt_cp' else 1)

X['cp_time'] = X['cp_time'].apply(lambda x: 0 if x==24 else 1)

X['cp_dose'] = X['cp_dose'].apply(lambda x: 0 if x=='D1' else 1)
preds = model.predict(X)
sub = dfs['sample_submission']

for i, p in zip(sub.index, preds):

    sub.loc[i, 1:] = p

sub.head(3)
sub.to_csv('submission.csv', index=False)