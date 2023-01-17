from IPython.display import Image

Image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Three_Phase_Electric_Power_Transmission.jpg/1200px-Three_Phase_Electric_Power_Transmission.jpg')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pyarrow.parquet as pq #reading parquet files 

import matplotlib.pyplot as plt

import os

import seaborn as sns
INIT_DIR = '../input'

SIZE = 2001
train = pq.read_pandas(os.path.join(INIT_DIR, 'vsb-power-line-fault-detection/train.parquet'), columns=[str(i) for i in range(SIZE)]).to_pandas()

metadata = pd.read_csv('../input/vsb-power-line-fault-detection/metadata_train.csv')
train.head()
train.shape
metadata.head()
metadata.shape
train_metadata = metadata[:SIZE]
train_metadata.shape
train = train.T
train.head(2)
train['signal_id'] = list(train_metadata['signal_id'])
train.head(2)
train = train.merge(train_metadata, on='signal_id')
train.head(2)
train.isnull().sum().sum()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

sns.countplot(x="target", data=train, ax=ax1)

sns.countplot(x="target", data=train, hue="phase", ax=ax2);
# https://www.w3resource.com/graphics/matplotlib/piechart/matplotlib-piechart-exercise-2.php

plt.rcParams["figure.figsize"] = (40,6.5)

data = train['target'].value_counts()

labels = ['Target 0', 'Target 1']

colors = ["#1f77b4", "#ff7f0e"]

title = 'Count of signals distributed by phase'

explodes = [0, 0.1]

plt.pie(data,explode=explodes, labels=labels, colors=colors, shadow=True, startangle=20, autopct='%.1f%%')

plt.title(title, bbox={'facecolor':'0.8', 'pad':5})

plt.show()
target_count = train.target.value_counts()

print("negative(target=0) target: {}".format(target_count[0]))

print("positive(target=1) target: {}".format(target_count[1]))

print("positive data {:.3}%".format((target_count[1]/(target_count[0]+target_count[1]))*100))
train[['id_measurement', 'phase']]
target_mismatch = train[["id_measurement", "target"]].groupby(["id_measurement"]).sum().query("target != 3 & target != 0")

print("Target values not all postive or negative for same signal: {}".format(target_mismatch.shape[0]))

target_mismatch
train[train['id_measurement'] == 67]
print("id_measurement have {} unique values".format(train.id_measurement.nunique()))
train.id_measurement.value_counts().describe()
print("phase have {} unique values {} in train".format(len(train.phase.unique()),train.phase.unique()))
sns.countplot(train['phase']);
# https://www.w3resource.com/graphics/matplotlib/piechart/matplotlib-piechart-exercise-2.php

data = train['phase'].value_counts()

labels = ['Phase 0', 'Phase 1', 'Phase 3']

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

title = 'Count of signals distributed by phase'

plt.pie(data, labels=labels, colors=colors, shadow=True, startangle=90, autopct='%.1f%%')

plt.title(title, bbox={'facecolor':'0.8', 'pad':5})

plt.show()
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)



X_embedding = tsne.fit_transform(train[:500])

y = np.array(train['target'][:500])



for_tsne = np.hstack((X_embedding, y.reshape(-1,1)))

for_tsne_df = pd.DataFrame(data=for_tsne, columns=['Dimension_x','Dimension_y','Score'])

colors = {0:'red', 1:'blue', 2:'green'}

plt.scatter(for_tsne_df['Dimension_x'], for_tsne_df['Dimension_y'], c=for_tsne_df['Score'].apply(lambda x: colors[x]))

plt.show()



del(tsne)
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, perplexity=50, learning_rate=200, random_state=42)



X_embedding = tsne.fit_transform(train[:500])

y = np.array(train['target'][:500])



for_tsne = np.hstack((X_embedding, y.reshape(-1,1)))

for_tsne_df = pd.DataFrame(data=for_tsne, columns=['Dimension_x','Dimension_y','Score'])

colors = {0:'red', 1:'blue', 2:'green'}

plt.scatter(for_tsne_df['Dimension_x'], for_tsne_df['Dimension_y'], c=for_tsne_df['Score'].apply(lambda x: colors[x]))

plt.show()



del(tsne)
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, perplexity=100, learning_rate=150, random_state=42)



X_embedding = tsne.fit_transform(train[:500])

y = np.array(train['target'][:500])



for_tsne = np.hstack((X_embedding, y.reshape(-1,1)))

for_tsne_df = pd.DataFrame(data=for_tsne, columns=['Dimension_x','Dimension_y','Score'])

colors = {0:'red', 1:'blue', 2:'green'}

plt.scatter(for_tsne_df['Dimension_x'], for_tsne_df['Dimension_y'], c=for_tsne_df['Score'].apply(lambda x: colors[x]))

plt.show()



del(tsne)
#signal with target 0 (normal signal)

train.loc[1]['target']
plt.figure(figsize=(24, 8))

plt.plot((train.loc[1].values), alpha=0.7);

plt.ylim([-100, 100])
#signal with target 1 (Faulty Signal)

train.loc[201]['target']
plt.figure(figsize=(24, 8))

plt.plot((train.loc[201].values), alpha=0.7);

plt.ylim([-100, 100])
#signal with target 0 (Normal Signal)

train.loc[0:2][['target', 'id_measurement']]
plt.figure(figsize=(24, 8))

plt.plot((train.loc[0].values), alpha=0.7);

plt.plot((train.loc[1].values), alpha=0.7);

plt.plot((train.loc[2].values), alpha=0.7);

plt.ylim([-100, 100])
#signal with target 1 (Faulty Signal)

train.loc[3:5][['target', 'id_measurement']]
plt.figure(figsize=(24, 8))

plt.plot((train.loc[3].values), alpha=0.7);

plt.plot((train.loc[4].values), alpha=0.7);

plt.plot((train.loc[5].values), alpha=0.7);

plt.ylim([-100, 100])
def flatiron(x, alpha=50, beta=1):

    new_x = np.zeros_like(x)

    zero = x[0]

    for i in range(1, len(x)):

        zero = zero*(alpha-beta)/alpha + beta*x[i]/alpha

        new_x[i] =  x[i] - zero

    return new_x
#Flattening a Normal signal

normal_sample_filt =  [None] * 3

normal_sample_filt[0] = flatiron(train.loc[0].values)

normal_sample_filt[1] = flatiron(train.loc[1].values)

normal_sample_filt[2] = flatiron(train.loc[2].values)
normal_sample_filt
#Code to plot faulty signal with flattened faulty signal

f, ax = plt.subplots(1, 2, figsize=(24, 8))



ax[0].plot((train.loc[0].values), alpha=0.7);

ax[0].plot((train.loc[1].values), alpha=0.7);

ax[0].plot((train.loc[2].values), alpha=0.7);

ax[0].set_title('Normal signal')

ax[0].set_ylim([-100, 100])



ax[1].plot((normal_sample_filt)[0], alpha=0.7);

ax[1].plot((normal_sample_filt)[1], alpha=0.7);

ax[1].plot((normal_sample_filt)[2], alpha=0.7);

ax[1].set_title('filtered Normal signal')

ax[1].set_ylim([-100, 100])



del(normal_sample_filt)
#Flattening a Faulty signal

fault_sample_filt =  [None] * 3

fault_sample_filt[0] = flatiron(train.loc[3].values)

fault_sample_filt[1] = flatiron(train.loc[4].values)

fault_sample_filt[2] = flatiron(train.loc[5].values)
fault_sample_filt
#Code to plot faulty signal with flattened faulty signal

f, ax = plt.subplots(1, 2, figsize=(24, 8))



ax[0].plot((train.loc[3].values), alpha=0.7);

ax[0].plot((train.loc[4].values), alpha=0.7);

ax[0].plot((train.loc[5].values), alpha=0.7);

ax[0].set_title('fault signal')

ax[0].set_ylim([-100, 100])



ax[1].plot((fault_sample_filt)[0], alpha=0.7);

ax[1].plot((fault_sample_filt)[1], alpha=0.7);

ax[1].plot((fault_sample_filt)[2], alpha=0.7);

ax[1].set_title('filtered fault signal')

ax[1].set_ylim([-100, 100])



del(fault_sample_filt)