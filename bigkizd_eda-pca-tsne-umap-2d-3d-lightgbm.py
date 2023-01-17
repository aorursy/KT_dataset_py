## Upgrade library

!/opt/conda/bin/python3.7 -m pip install --upgrade pip

!pip install -U seaborn
import os

import numpy as np

import pandas as pd



# general for visualization

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
def init_seed(SEED=42):

    os.environ['PYTHONHASHSEED'] = str(SEED)

    np.random.seed(SEED)

init_seed(42)
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_target = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

sub= pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



# concat train and target

df = train.set_index(['sig_id']).join(train_target.set_index('sig_id'))

(train.shape, train_target.shape, test.shape)
train.head(10)
test.head(10)
train_target.head(10)
# Top 10 missing values

fig, axes = plt.subplots(2, 2, figsize=(15, 8))

(train.isna().sum().sort_values()*100/len(train)).head(10).plot.barh(ax = axes[0][0])

(test.isna().sum().sort_values()*100/len(train)).head(10).plot.barh(ax = axes[0][1])

(train_target.isna().sum().sort_values()*100/len(train)).head(10).plot.barh(ax = axes[1][0])
fig, axes = plt.subplots(1, 3, figsize=(15, 8))

sns.countplot(data=train, x='cp_type', ax=axes[0])

sns.countplot(data=train, x='cp_dose', ax=axes[1])

sns.countplot(data=train, x='cp_time', ax=axes[2])



axes[0].set_title('Sample Treatment', fontsize=15)

axes[1].set_title('Dose Treatment', fontsize=15)

axes[2].set_title('Time Treatment', fontsize=15)
train['g_mean'] = train[[x for x in train.columns if x.startswith('g-')]].mean(axis=1)

sns.displot(data=train, x = f'g_mean', height=6, aspect=2)

print('Description of g-mean: ', train['g_mean'].describe())
sns.displot(data=train, x = f'g_mean', hue='cp_type', height=6, aspect=2)

sns.displot(data=train, x = f'g_mean', hue='cp_dose', height=6, aspect=2)

sns.displot(data=train, x = f'g_mean', hue='cp_time', height=6, aspect=2)
train['c_mean'] = train[[x for x in train.columns if x.startswith('c-')]].mean(axis=1)

sns.displot(data=train, x = 'c_mean', height=6, aspect=2)

print(train['c_mean'].describe())
sns.displot(data=train, x = f'c_mean', hue='cp_type', height=6, aspect=2)

sns.displot(data=train, x = f'c_mean', hue='cp_dose', height=6, aspect=2)

sns.displot(data=train, x = f'c_mean', hue='cp_time', height=6, aspect=2)
sns.countplot(data=df, x='cp_type', hue='trpv_agonist')
fig, axes = plt.subplots(1, 1, figsize=(50, 30))

cols = [x for x in train.columns if x.startswith('g-')]

sns.heatmap(data=train.drop(columns=['cp_type', 'cp_dose', 'sig_id'])[cols[:30]].corr(), ax=axes)


cols = [x for x in train.columns if x.startswith('g-')]

sns.clustermap(data=train.drop(columns=['cp_type', 'cp_dose', 'sig_id'])[cols[:30]].corr())
#PCA

from sklearn.decomposition import PCA

#TSNE

from sklearn.manifold import TSNE

#UMAP

from umap import UMAP
import warnings

warnings.filterwarnings('ignore')

g_cols = [x for x in train.columns if x.startswith('g-')]



pca = PCA(n_components=3).fit_transform(train[g_cols])

print('Done PCA')

pca_fake = PCA(n_components=10)

pca_result = pca_fake.fit_transform(train[g_cols])

print('Done PCA results')

tsne = TSNE(n_components=3).fit_transform(pca_result)

print('Done TSNE')

umap = UMAP(random_state=42,n_components=3).fit_transform(pca_result)

print('Done UMAP')



fig, axes = plt.subplots(3, 1, figsize=(12, 16))

sns.scatterplot(x=pca[:, 0], y=pca[:, 1], ax=axes[0])

sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], ax=axes[1])

sns.scatterplot(x=umap[:, 0], y=umap[:, 1], ax=axes[2])

ax = plt.figure(figsize=(10,8)).gca(projection='3d')

ax.scatter(

    xs=pca[:, 0], 

    ys=pca[:, 1], 

    zs=pca[:, 2], 

    cmap='gist_rainbow'

)

ax.set_xlabel('Principal Component 1')

ax.set_ylabel('Principal Component 2')

ax.set_zlabel('Principal Component 3')

plt.title('Visualizing g-feature PCA in 3D', fontsize=24);

plt.show()
ax = plt.figure(figsize=(10,8)).gca(projection='3d')

ax.scatter(

    xs=tsne[:, 0], 

    ys=tsne[:, 1], 

    zs=tsne[:, 2], 

    cmap='gist_rainbow'

)

ax.set_xlabel('TSNE 1')

ax.set_ylabel('TSNE 2')

ax.set_zlabel('TSNE 3')

plt.title('Visualizing g-feature TSNE in 3D', fontsize=24);

plt.show()
ax = plt.figure(figsize=(10,8)).gca(projection='3d')

ax.scatter(

    xs=umap[:, 0], 

    ys=umap[:, 1], 

    zs=umap[:, 2], 

    cmap='gist_rainbow'

)

ax.set_xlabel('UMAP 1')

ax.set_ylabel('UMAP 2')

ax.set_zlabel('UMAP 3')

plt.title('Visualizing g-feature UMAP in 3D', fontsize=24);

plt.show()
c_cols = [x for x in train.columns if x.startswith('c-')]



pca = PCA(n_components=3).fit_transform(train[c_cols])

print('Done PCA')

pca_fake = PCA(n_components=10)

pca_result = pca_fake.fit_transform(train[g_cols])

print('Done PCA results')

tsne = TSNE(n_components=3).fit_transform(pca_result)

print('Done TSNE')

umap = UMAP(random_state=42,n_components=3).fit_transform(pca_result)

print('Done UMAP')



fig, axes = plt.subplots(3, 1, figsize=(12, 16))

sns.scatterplot(x=pca[:, 0], y=pca[:, 1], ax=axes[0])

sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], ax=axes[1])

sns.scatterplot(x=umap[:, 0], y=umap[:, 1], ax=axes[2])

ax = plt.figure(figsize=(10,8)).gca(projection='3d')

ax.scatter(

    xs=pca[:, 0], 

    ys=pca[:, 1], 

    zs=pca[:, 2], 

    cmap='gist_rainbow'

)

ax.set_xlabel('Principal Component 1')

ax.set_ylabel('Principal Component 2')

ax.set_zlabel('Principal Component 3')

plt.title('Visualizing c-feature PCA in 3D', fontsize=24);

plt.show()
ax = plt.figure(figsize=(10,8)).gca(projection='3d')

ax.scatter(

    xs=tsne[:, 0], 

    ys=tsne[:, 1], 

    zs=tsne[:, 2], 

    cmap='gist_rainbow'

)

ax.set_xlabel('TSNE 1')

ax.set_ylabel('TSNE 2')

ax.set_zlabel('TSNE 3')

plt.title('Visualizing c-feature TSNE in 3D', fontsize=24);

plt.show()
ax = plt.figure(figsize=(10,8)).gca(projection='3d')

ax.scatter(

    xs=umap[:, 0], 

    ys=umap[:, 1], 

    zs=umap[:, 2], 

    cmap='gist_rainbow'

)

ax.set_xlabel('UMAP 1')

ax.set_ylabel('UMAP 2')

ax.set_zlabel('UMAP 3')

plt.title('Visualizing c-feature UMAP in 3D', fontsize=24);

plt.show()
train_target.drop(columns=['sig_id']).sum().sort_values(ascending=False).head(30).plot.barh(figsize=(8, 8))
train_target.drop(columns=['sig_id']).sum().sort_values(ascending=True).head(30).plot.barh(figsize=(8, 8))
train['cp_type'] = train['cp_type'].replace({'trt_cp': 0, "ctl_vehicle": 1})

train['cp_dose'] = train['cp_dose'].replace({'D1': 0, "D2": 1})



test['cp_type'] = test['cp_type'].replace({'trt_cp': 0, "ctl_vehicle": 1})

test['cp_dose'] = test['cp_dose'].replace({'D1': 0, "D2": 1})
column_X = train.drop(columns=['sig_id']).columns

column_y = train_target.drop(columns=['sig_id']).columns
datatrain = train.set_index(['sig_id']).join(train_target.set_index('sig_id'))


## submission

sub['sig_id'] = test['sig_id']

params = {

    "task": 'train',

    "boosting_type": 'gbdt',

    "num_leaves": 128,

    "max_depth": 20,

    "n_estimators": 150,

    "metrics": "auc"



}



data_train, data_val = train_test_split(datatrain, test_size=0.2)

for col in tqdm(column_y):

    train_lgb = lgb.Dataset(data_train[column_X], data_train[col])

    val_lgb = lgb.Dataset(data_val[column_X], data_val[col])

    clf = lgb.train(params = params, train_set=train_lgb, valid_sets=val_lgb, verbose_eval=50)

    sub[col] = np.where(clf.predict(test.drop(columns=['sig_id']))>0.5, 1, 0)

    

    print("{} Done^^".format(col))

    break

print('CPU chan qua=)) To be continue...')
