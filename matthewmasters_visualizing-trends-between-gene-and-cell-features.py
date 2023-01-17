import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def plot_features(ax, features, color='blue', title=''):
    ax.scatter(np.arange(len(features)), features, c=color)
    ax.set_ylabel('Feature Value')
    ax.set_xlabel('Feature #')
    ax.set_title(title)
    
def plot_one_vs_class(ax, idx, class_features, title=''):
    for i, feat in enumerate(class_features):
        if i == idx: continue
        plot_features(ax, feat, 'grey')
    
    plot_features(ax, class_features[idx], title=title)
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_ns = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
target = 'radiopaque_medium'
target_df = train_features[train_targets[target] == 1].reset_index(drop=True)
target_features = target_df.values[:, 4:]
fig, ax = plt.subplots(2, 1, figsize=(20, 10))
plot_one_vs_class(ax[0], 0, target_features)
plot_one_vs_class(ax[1], 1, target_features)
target = 'igf-1_inhibitor'
target_df = train_features[train_targets[target] == 1].reset_index(drop=True)
target_features = target_df.values[:, 4:]
fig, ax = plt.subplots(2, 1, figsize=(20, 10))
plot_one_vs_class(ax[0], 6, target_features)
plot_one_vs_class(ax[1], 2, target_features)
x = np.mean(train_features.values[:, -100:], axis=1)
y = np.std(train_features.values[:, 4:-100].astype(np.float), axis=1)
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.scatter(x, y, c='blue')
ax.set_xlabel('Average Cell Viability')
ax.set_ylabel('Standard Deviation of Gene Expression')
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(20, 5))
plot_features(ax, train_features[y > 7].values[0, 4:])
targets = train_targets.columns[1:][train_targets[y > 7].values[0, 1:].astype(np.bool)]
print('\n'.join(targets))
target = 'proteasome_inhibitor'
data = x[train_targets[target] == 1]
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.hist(data, 20, color='blue')
ax.set_xlabel('Average Cell Viability')
ax.set_ylabel('Count')
ax.set_xlim(-10, 10)
ax.set_title(target)
plt.show()