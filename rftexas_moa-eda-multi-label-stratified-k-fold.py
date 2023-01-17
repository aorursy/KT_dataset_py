!pip install iterative-stratification
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



import random



from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



from sklearn.manifold import TSNE

from sklearn.decomposition import PCA



from sklearn.preprocessing import LabelEncoder
DATA_PATH = '../input/lish-moa/'



TRAIN_FEATURES = DATA_PATH + 'train_features.csv'

TEST_FEATURES = DATA_PATH + 'test_features.csv'

TRAIN_TARGETS_NON_SCORED = DATA_PATH + 'train_targets_nonscored.csv'

TRAIN_TARGETS_SCORED = DATA_PATH + 'train_targets_scored.csv'
train_features_df = pd.read_csv(TRAIN_FEATURES)

train_targets_df = pd.read_csv(TRAIN_TARGETS_SCORED)
train_features_df.head()
print(f'There are {len(train_features_df)} samples.')

print(f'There are {len(train_features_df["sig_id"].unique())} unique samples.')

print(f'There are {len(train_features_df.columns)-1} predictive features.')

print(f'There are {(train_features_df.dtypes == "float64").sum()} continuous features.')

print(f'There are {len(train_targets_df.columns)-1} classes.')

print(f'{train_features_df.isna().sum().sum()} missing value')
temp = train_features_df.groupby('cp_type').count()['sig_id'].reset_index().sort_values(by='sig_id',ascending=False)

temp.style.background_gradient(cmap='Reds')
temp = train_features_df.groupby('cp_time').count()['sig_id'].reset_index().sort_values(by='sig_id',ascending=False)

temp.style.background_gradient(cmap='Blues')
temp = train_features_df.groupby('cp_dose').count()['sig_id'].reset_index().sort_values(by='sig_id',ascending=False)

temp.style.background_gradient(cmap='Greens')
fig, ax = plt.subplots(1, 3, figsize=(20, 5))



sns.countplot(train_features_df['cp_type'], ax=ax[0])

ax[0].set_title('cp_type distribution')



sns.countplot(train_features_df['cp_time'], ax=ax[1])

ax[1].set_title('cp_time distribution')



sns.countplot(train_features_df['cp_dose'], ax=ax[2])

ax[2].set_title('cp_dose distribution')



fig.suptitle('Distribution of type, time and dose')
gene_features = list([x for x in list(train_features_df.columns) if "g-" in x])



print(f'There are {len(gene_features)} gene features.')
fig, ax = plt.subplots(5, 5, figsize=(20, 25))

rand_feats = random.choices(gene_features, k=25)



fig.suptitle(

    'Some gene features distribution - Just to have a look...', 

    fontsize=15, 

    fontweight='bold'

)



for x in range(25):

    i = x // 5

    j = x % 5

    

    sns.distplot(train_features_df[rand_feats[x]], ax=ax[i][j])

    ax[i][j].set_title(rand_feats[x])
mean_stats = train_features_df[gene_features].mean()

std_stats = train_features_df[gene_features].std()

skew_stats = train_features_df[gene_features].skew()

kurt_stats = train_features_df[gene_features].kurt()
fig, ax = plt.subplots(2, 2, figsize=(10, 10))



sns.distplot(

    mean_stats, 

    kde_kws=

        {

            "color": "blue", 

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "blue"

        },

    ax=ax[0][0])

ax[0][0].set_title('Mean')



sns.distplot(

    std_stats, 

    kde_kws=

        {

            "color": "red", 

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "red"

        },

    ax=ax[0][1])

ax[0][1].set_title('Standard deviation')



sns.distplot(

    skew_stats, 

    kde_kws=

        {

            "color": "green", 

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "green"

        },

    ax=ax[1][0]

)

ax[1][0].set_title('Skew')



sns.distplot(

    kurt_stats,

    kde_kws=

        {

            "color": "orange", 

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "orange"

        },

    ax=ax[1][1])

ax[1][1].set_title('Kurtosis')



fig.suptitle('Gene features - Metastatistics distribution')
cell_features = list([x for x in list(train_features_df.columns) if "c-" in x])



print(f'There are {len(cell_features)} gene features.')
fig, ax = plt.subplots(5, 5, figsize=(20, 25))

rand_feats = random.choices(cell_features, k=25)



fig.suptitle(

    'Some gene features distribution - Just to have a look...', 

    fontsize=15, 

    fontweight='bold'

)



for x in range(25):

    i = x // 5

    j = x % 5

    

    sns.distplot(train_features_df[rand_feats[x]], ax=ax[i][j])

    ax[i][j].set_title(rand_feats[x])
mean_stats = train_features_df[cell_features].mean()

std_stats = train_features_df[cell_features].std()

skew_stats = train_features_df[cell_features].skew()

kurt_stats = train_features_df[cell_features].kurt()
fig, ax = plt.subplots(2, 2, figsize=(10, 10))



sns.distplot(

    mean_stats, 

    kde_kws=

        {

            "color": "blue", 

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "blue"

        },

    ax=ax[0][0])

ax[0][0].set_title('Mean')



sns.distplot(

    std_stats, 

    kde_kws=

        {

            "color": "red", 

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "red"

        },

    ax=ax[0][1])

ax[0][1].set_title('Standard deviation')



sns.distplot(

    skew_stats, 

    kde_kws=

        {

            "color": "green", 

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "green"

        },

    ax=ax[1][0]

)

ax[1][0].set_title('Skew')



sns.distplot(

    kurt_stats,

    kde_kws=

        {

            "color": "orange", 

        },

    hist_kws=

        {

            "linewidth": 3,

            "color": "orange"

        },

    ax=ax[1][1])

ax[1][1].set_title('Kurtosis')



fig.suptitle('Cell features - Metastatistics distribution')
fig = plt.figure(figsize=(10,6))

ax = sns.countplot(x="cp_time", hue="cp_dose", data=train_features_df)



for p in ax.patches:

    '''

    https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline

    '''

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2,

                height +3,

                '{:1.2f}%'.format(100*height/len(train_features_df)),

                ha="center")



ax.set_title('Distribution of cp_time with respect to dose')
fig = plt.figure(figsize=(10,6))

ax = sns.countplot(x="cp_type", hue="cp_dose", data=train_features_df)



for p in ax.patches:

    '''

    https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline

    '''

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2,

                height +3,

                '{:1.2f}%'.format(100*height/len(train_features_df)),

                ha="center")



ax.set_title('Distribution of cp_time with respect to dose')
fig = plt.figure(figsize=(10,6))

ax = sns.countplot(x="cp_time", hue="cp_type", data=train_features_df)



for p in ax.patches:

    '''

    https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline

    '''

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2,

                height +3,

                '{:1.2f}%'.format(100*height/len(train_features_df)),

                ha="center")



ax.set_title('Distribution of cp_time with respect to dose')
train_targets_df.head()
# Plot of multiple labels for one id



target_cols = list(train_targets_df.columns)

target_cols.remove('sig_id')



multiple_labels = train_targets_df[target_cols].sum(axis=1)



fig, ax = plt.subplots(1, 1, figsize=(10, 5))

sns.countplot(multiple_labels, ax=ax)

ax.set_title('Distribution of number of labels')
multiple_labels = train_targets_df[target_cols].sum(axis=0).sort_values(ascending=False)[:5]



fig, ax = plt.subplots(1, 1, figsize=(10, 5))

chart = sns.barplot(multiple_labels.index, multiple_labels.values)

ax.set_title('Most frequent triggered mechanisms')



chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
train_features_df['cp_time'] = train_features_df['cp_time'].map({24: 0, 48: 1, 72: 2})

train_features_df['cp_dose'] = train_features_df['cp_dose'].map({'D1': 3, 'D2': 4})
cat_features = ['cp_type', 'cp_time', 'cp_dose']

features = cell_features +  gene_features + cat_features



X = train_features_df[features].values

y_1 = train_targets_df[target_cols].sum(axis=1).values

y_2 = train_targets_df['nfkb_inhibitor'].values

y_3 = train_targets_df['proteasome_inhibitor'].values

y_4 = train_targets_df['cyclooxygenase_inhibitor'].values



indices = random.choices(range(len(X)), k=2000)



X = X[indices,]

y_1 = y_1[indices,]

y_2 = y_2[indices,]

y_3 = y_3[indices,]

y_4 = y_4[indices,]



print('X shape:', X.shape)

print('y shape:', y_1.shape)
pca = PCA(n_components=50)

X_reduced = pca.fit_transform(X)



t_sne_results_2d = TSNE(n_components=2).fit_transform(X_reduced)
fig, ax = plt.subplots(1, 1, figsize=(16,10))

sns.scatterplot(

    x=t_sne_results_2d[:, 0], 

    y=t_sne_results_2d[:, 1],

    hue=y_1,

    palette=sns.color_palette("hls", 6),

    legend="full",

    alpha=0.3,

    ax=ax

)



ax.set_title('2d visualization of T-SNE components  - Number of labels')
fig, ax = plt.subplots(1, 1, figsize=(16,10))

sns.scatterplot(

    x=t_sne_results_2d[:, 0], 

    y=t_sne_results_2d[:, 1],

    hue=y_2,

    palette=sns.color_palette("hls", 2),

    legend="full",

    alpha=0.3,

    ax=ax

)



ax.set_title('2d visualization of T-SNE components - nfkb_inhibitor triggered ')
fig, ax = plt.subplots(1, 1, figsize=(16,10))

sns.scatterplot(

    x=t_sne_results_2d[:, 0], 

    y=t_sne_results_2d[:, 1],

    hue=y_3,

    palette=sns.color_palette("hls", 2),

    legend="full",

    alpha=0.3,

    ax=ax

)



ax.set_title('2d visualization of T-SNE components - proteasome_inhibitor triggered')
fig, ax = plt.subplots(1, 1, figsize=(16,10))

sns.scatterplot(

    x=t_sne_results_2d[:, 0], 

    y=t_sne_results_2d[:, 1],

    hue=y_4,

    palette=sns.color_palette("hls", 2),

    legend="full",

    alpha=0.3,

    ax=ax

)



ax.set_title('2d visualization of T-SNE components - cyclooxygenase_inhibitor')
kfold = MultilabelStratifiedKFold(n_splits=5)



X = train_features_df[features]

y = train_targets_df[target_cols]



full_df = train_features_df.merge(train_targets_df, on="sig_id", how='left')



for i, (trn_, val_) in enumerate(kfold.split(X, y)):

    full_df.loc[val_, 'fold'] = i
print('Shape:', full_df.shape)
full_df.to_csv('df.csv', index=False)