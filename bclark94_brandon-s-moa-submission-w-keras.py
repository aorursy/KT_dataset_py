

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow_addons as tfa
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from tqdm.notebook import tqdm
%matplotlib inline
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
train_features.sample(5)
test_features.sample(5)
train_targets_scored.sample(5)
train_features['dataset'] = 'train'
test_features['dataset'] = 'test'

df = pd.concat([train_features, test_features])

ds = df.groupby(['cp_type', 'dataset'])['sig_id'].count().reset_index()
ds.columns = ['cp_type', 'dataset', 'count']

fig = px.bar(
    ds, 
    x='cp_type', 
    y="count", 
    color = 'dataset',
    barmode='group',
    orientation='v', 
    title='cp_type train/test counts', 
    width=500,
    height=400
)

fig.show()
ds = df.groupby(['cp_time', 'dataset'])['sig_id'].count().reset_index()
ds.columns = ['cp_time', 'dataset', 'count']

fig = px.bar(
    ds, 
    x='cp_time', 
    y="count", 
    color = 'dataset',
    barmode='group',
    orientation='v', 
    title='cp_time train/test counts', 
    width=500,
    height=400
)

fig.show()
ds = df.groupby(['cp_dose', 'dataset'])['sig_id'].count().reset_index()
ds.columns = ['cp_dose', 'dataset', 'count']

fig = px.bar(
    ds, 
    x='cp_dose', 
    y="count", 
    color = 'dataset',
    barmode='group',
    orientation='v', 
    title='cp_dose train/test counts', 
    width=500,
    height=400
)

fig.show()
#gene data distribution
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, column in enumerate(train_features[train_features.columns[4:20]].columns):
    sns.distplot(train_features[column], ax=axes[i // 4, i % 4])
plt.tight_layout()
plt.show()
#cell data distribution
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, column in enumerate(train_features[train_features.columns[776:792]].columns):
    sns.distplot(train_features[column], ax=axes[i // 4, i % 4], color='green')
plt.tight_layout()
plt.show()
correlations = train_features.corr()
# plot correlation matrix
fig = plt.figure(figsize=(19, 17))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
plt.show()
correlations.sample(5)
x = train_targets_scored.drop(['sig_id'], axis=1)
x = x.T
x["sum"] = x.sum(axis=1)
x = x['sum'].sort_values().reset_index()
x.columns = ['column','nonzero_records']

fig = px.bar(
    x.tail(50), 
    x='nonzero_records', 
    y='column', 
    orientation='h', 
    title='Columns with the most number of positive samples (top 50)', 
    height=1000, 
    width=800
)

fig.show()
fig = px.bar(
    x.head(50), 
    x='nonzero_records', 
    y='column', 
    orientation='h', 
    title='Columns with least number of positive samples (top 50)', 
    height=1000, 
    width=800
)

fig.show()
data = train_targets_scored.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()
data.columns = ['row', 'count']
data = data.groupby(['count'])['row'].count().reset_index()

fig = px.bar(
    data, 
    y=data['row'], 
    x="count", 
    title='Number of activations in targets for every sample', 
    width=800, 
    height=500
)

fig.show()
train_targets_scored.describe()
target_columns = list(train_targets_scored.columns) 

last_term = dict()
for item in target_columns:
    try:
        last_term[item.split('_')[-1]] += 1
    except:
        last_term[item.split('_')[-1]] = 1

last_term = pd.DataFrame(last_term.items(), columns=['group', 'count'])
last_term = last_term.sort_values('count')
last_term = last_term[last_term['count']>1]
last_term['count'] = last_term['count'] * 100 / 206

fig = px.bar(
    last_term, 
    x='count', 
    y="group", 
    orientation='h', 
    title='Groups in target columns (Percent from all target columns)', 
    width=800,
    height=500
)

fig.show()
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
def preprocess(df):
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1}) #change cp_type to binary 0,1 output to predict
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1}) #change cp_dose to binary 0,1 output to predict
    del df['sig_id'] #remove unique identifer so that entire data frame is numeric features
    return df

train = preprocess(train_features)
test = preprocess(test_features)

train_targets = train_targets_scored.copy()
del train_targets['sig_id']

train_targets = train_targets.loc[train['cp_type']==0].reset_index(drop=True) #remove control datapoints from train targets
train = train.loc[train['cp_type']==0].reset_index(drop=True) #remove control datapoints from train observations
def create_model(num_columns):
    model = tf.keras.Sequential([
    tf.keras.layers.Input(num_columns),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tfa.layers.WeightNormalization(tf.keras.layers.Dense(1048, activation="relu")),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation="sigmoid"))
    ])
    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),
                  loss='binary_crossentropy', 
                  )
    return model
def metric(y_true, y_pred):
    metrics = []
    for _target in train_targets.columns:
        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels=[0,1]))
    return np.mean(metrics)
num_columns = len(train.columns)
model = create_model(num_columns)

history = model.fit(train[2000:],
                    train_targets[2000:],
                    validation_data=(train[:2000], train_targets[:2000]),
                    epochs=30
                    )
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,.05)
plt.show()
submission.loc[:, train_targets.columns] = 0 #set all values in submission to 0.  Will be replaced with predicted values from model

res = train_targets.copy()
res.loc[:, train_targets.columns] = 0
res = res[:2000]

test_predict = model.predict(test.values)
val_predict = model.predict(train.values[:2000])


submission.sample(5)
submission.loc[:, train_targets.columns] += test_predict
res.loc[:, train_targets.columns] += val_predict
#res.loc[te, train_targets.columns] += val_predict
print(f'OOF Metric: {metric(train_targets[:2000], res)}')

submission.loc[test['cp_type']==1, train_targets.columns] = 0
submission.to_csv('submission.csv', index=False)
submission.head()