# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install /kaggle/input/whl-datasets/iterative_stratification-0.1.6-py3-none-any.whl



from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



import seaborn as sns

from numpy import mean, std

import seaborn as sns

from matplotlib import *

from matplotlib import pyplot as plt

import matplotlib.patches as mpatches



import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import tensorflow_addons as tfa

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

from tqdm.notebook import tqdm



import warnings

warnings.simplefilter("ignore")
trainFeatures = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

testFeatures = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

trainTargetsS = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

trainTargetsNS = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



for file in (trainFeatures, testFeatures, trainTargetsS, trainTargetsNS):

    file.columns = file.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    

print('Train features shape: ', trainFeatures.shape)

print('Test features shape: ', testFeatures.shape)

print('Train Targets (Scored) shape: ', trainTargetsS.shape)

print('Train Targets (Non Scored) shape: ', trainTargetsNS.shape)



print('------- Train Features view -------')

trainFeatures.head()
print('------- Train Targets (Scored) view -------')

trainTargetsS.head()
print('------- Train Targets (Non Scored) view -------')

trainTargetsNS.head()
# The Sample Treatment is heavily skewed:



print('Compound Treatment', round(trainFeatures['cp_type'].value_counts()[0]/len(trainFeatures) * 100, 2), '% of the dataset')

print('Control Perturbation Treatment', round(trainFeatures['cp_type'].value_counts()[1]/len(trainFeatures) * 100, 2), '% of the dataset')
colors = ["#0101DF", "#DF0101"]



sns.countplot('cp_type', data = trainFeatures, palette = colors)

plt.title('Sample Treatment Distribution \n (trt_cp: Compound Treatment || clt_vehicle: Control Perturbation Treatment)', fontsize = 14)

plt.show()
# The Treatment Dose is almost equally distributed:



print('High Dose', round(trainFeatures['cp_dose'].value_counts()[0]/len(trainFeatures) * 100, 2), '% of the dataset')

print('Low Dose', round(trainFeatures['cp_dose'].value_counts()[1]/len(trainFeatures) * 100, 2), '% of the dataset')



colors = ["#0101DF", "#DF0101"]



sns.countplot('cp_dose', data = trainFeatures, palette = colors)

plt.title('Treatment Dose Distribution \n (High Dose || Low Dose)', fontsize = 14)

plt.show()
# The Treatment Duration is almost equally distributed:



duration = pd.DataFrame(trainFeatures['cp_time'].value_counts()).reset_index()



print('24 hrs. of treatment doses', round(duration.loc[0][1]/len(trainFeatures) * 100, 2), '% of the dataset')

print('48 hrs. of treatment doses', round(duration.loc[1][1]/len(trainFeatures) * 100, 2), '% of the dataset')

print('72 hrs. of treatment doses', round(duration.loc[2][1]/len(trainFeatures) * 100, 2), '% of the dataset')



colors = ["#0101DF", "#DF0101", "#008000"]



sns.countplot('cp_time', data = trainFeatures, palette = colors)

plt.title('Treatment Duration Distribution \n (24 || 48 || 72 units of hours)', fontsize = 14)

plt.show()
fig, ax = plt.subplots(1, 4, figsize = (22, 4))



gene3 = trainFeatures['g-3'].values

gene2 = trainFeatures['g-2'].values

gene1 = trainFeatures['g-1'].values

gene0 = trainFeatures['g-0'].values



sns.distplot(gene3, ax = ax[0], color = 'r')

ax[0].set_title('Distribution of gene3', fontsize = 14)

ax[0].set_xlim([min(gene3), max(gene3)])



sns.distplot(gene2, ax = ax[1], color = 'b')

ax[1].set_title('Distribution of gene2', fontsize = 14)

ax[1].set_xlim([min(gene2), max(gene2)])



sns.distplot(gene1, ax = ax[2], color = 'g')

ax[2].set_title('Distribution of gene1', fontsize = 14)

ax[2].set_xlim([min(gene1), max(gene1)])



sns.distplot(gene0, ax = ax[3], color = 'y')

ax[3].set_title('Distribution of gene0', fontsize = 14)

ax[3].set_xlim([min(gene0), max(gene0)])



plt.show()
Skewness = pd.DataFrame(trainFeatures.skew()).reset_index()

Skewness.columns = ['column', 'skewness']

Skewness = Skewness.sort_values('skewness')

Skewness.head()
cell_cols = [col for col in trainFeatures if col.startswith('c-')]

gene_cols = [col for col in trainFeatures if col.startswith('g-')]



cellMeans = trainFeatures[cell_cols].mean()

cellMeans = pd.DataFrame(cellMeans).reset_index()

cellMeans.columns = ['column', 'mean']

geneMeans = trainFeatures[gene_cols].mean()

geneMeans = pd.DataFrame(geneMeans).reset_index()

geneMeans.columns = ['column', 'mean']





fig, ax = plt.subplots(1, 2, figsize = (22, 4))



cell = cellMeans['mean'].values

gene = geneMeans['mean'].values



sns.distplot(cell, ax = ax[0], color = 'r')

ax[0].set_title('Distribution of means of cell variables', fontsize = 14)

ax[0].set_xlim([min(cell), max(cell)])



sns.distplot(gene, ax = ax[1], color = 'b')

ax[1].set_title('Distribution of means of gene variables', fontsize = 14)

ax[1].set_xlim([min(gene), max(gene)])



plt.show()
cellMins = trainFeatures[cell_cols].max()

cellMins = pd.DataFrame(cellMins).reset_index()

cellMins.columns = ['column', 'MAX']

geneMins = trainFeatures[gene_cols].max()

geneMins = pd.DataFrame(geneMins).reset_index()

geneMins.columns = ['column', 'MAX']





fig, ax = plt.subplots(1, 2, figsize = (22, 4))



cell = cellMins['MAX'].values

gene = geneMins['MAX'].values



sns.distplot(cell, ax = ax[0], color = 'r')

ax[0].set_title('Distribution of maximums of cell variables', fontsize = 14)

ax[0].set_xlim([min(cell), max(cell)])



sns.distplot(gene, ax = ax[1], color = 'b')

ax[1].set_title('Distribution of maximums of gene variables', fontsize = 14)

ax[1].set_xlim([min(gene), max(gene)])



plt.show()
plt.figure(figsize = (20, 6))

targetTypes = []

for column in trainTargetsS.columns:

    try:

        targetTypes.append(column.rsplit('_', 1)[1])

    except:

        targetTypes.append(column.rsplit('_', 1)[0])

targetTypes = list(dict.fromkeys(targetTypes))



targetTypes.remove('id')

targetTypes.remove('b')



targets = {}

for i in targetTypes:

    targets[i] = 0



for column in trainTargetsS.columns:

    try:

        col = column.rsplit('_', 1)[1]

        if col not in ['id', 'b']:

            targets[col] += 1

        else: pass

                

    except:

        col = column.rsplit('_', 1)[0]

        if col not in ['id', 'b']:

            targets[col] += 1

        else: pass

targets = pd.DataFrame.from_dict(targets, orient = 'index')

targets = targets.reset_index()

targets.columns = ['target_type', 'types']

sns.barplot(x = 'target_type', y = 'types', data = targets)

plt.xticks(rotation = 90)

plt.title('Types of Target variables')

plt.show()
correlations = trainFeatures.corr()

correlationsM = correlations.abs()

kot = correlations[(correlationsM >= .9) & (correlationsM != 1)]

kot = kot.dropna(axis = 1, how = 'all')

kot = kot.dropna(axis = 0, how = 'all')

plt.figure(figsize = (20, 20))

sns.heatmap(kot, cmap = "Greens")
def preprocess(df):

    df = df.copy()

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



train = preprocess(trainFeatures)

test = preprocess(testFeatures)



del trainTargetsS['sig_id']



trainTargetsS = trainTargetsS.loc[train['cp_type'] == 0].reset_index(drop = True)

train = train.loc[train['cp_type'] == 0].reset_index(drop = True)
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

    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period = 10),

                  loss = 'binary_crossentropy',

                  )

    return model
seed = 42



from typing import Tuple, List, Callable, Any



from sklearn.utils import check_random_state  # type: ignore



### from eli5

def iter_shuffled(X, columns_to_shuffle=None, pre_shuffle=False,

                  random_state=None):

    rng = check_random_state(random_state)



    if columns_to_shuffle is None:

        columns_to_shuffle = range(X.shape[1])



    if pre_shuffle:

        X_shuffled = X.copy()

        rng.shuffle(X_shuffled)



    X_res = X.copy()

    for columns in tqdm(columns_to_shuffle):

        if pre_shuffle:

            X_res[:, columns] = X_shuffled[:, columns]

        else:

            rng.shuffle(X_res[:, columns])

        yield X_res

        X_res[:, columns] = X[:, columns]







def get_score_importances(

        score_func,  # type: Callable[[Any, Any], float]

        X,

        y,

        n_iter=5,  # type: int

        columns_to_shuffle=None,

        random_state=None

    ):

    rng = check_random_state(random_state)

    base_score = score_func(X, y)

    scores_decreases = []

    for i in range(n_iter):

        scores_shuffled = _get_scores_shufled(

            score_func, X, y, columns_to_shuffle=columns_to_shuffle,

            random_state=rng, base_score=base_score

        )

        scores_decreases.append(scores_shuffled)



    return base_score, scores_decreases







def _get_scores_shufled(score_func, X, y, base_score, columns_to_shuffle=None,

                        random_state=None):

    Xs = iter_shuffled(X, columns_to_shuffle, random_state=random_state)

    res = []

    for X_shuffled in Xs:

        res.append(-score_func(X_shuffled, y) + base_score)

    return res



def metric(y_true, y_pred):

    metrics = []

    for i in range(y_pred.shape[1]):

        if y_true[:, i].sum() > 1:

            metrics.append(log_loss(y_true[:, i], y_pred[:, i].astype(float)))

    return np.mean(metrics)   



perm_imp = np.zeros(train.shape[1])

all_res = []

for n, (tr, te) in enumerate(KFold(n_splits=7, random_state=0, shuffle=True).split(trainTargetsS)):

    print(f'Fold {n}')



    model = create_model(len(train.columns))

    checkpoint_path = f'repeat:{seed}_Fold:{n}.hdf5'

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

    cb_checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True,

                                     save_weights_only = True, mode = 'min')

    model.fit(train.values[tr],

                  trainTargetsS.values[tr],

                  validation_data=(train.values[te], trainTargetsS.values[te]),

                  epochs=35, batch_size=128,

                  callbacks=[reduce_lr_loss, cb_checkpt], verbose=2

                 )

        

    model.load_weights(checkpoint_path)

        

    def _score(X, y):

        pred = model.predict(X)

        return metric(y, pred)



    base_score, local_imp = get_score_importances(_score, train.values[te], trainTargetsS.values[te], n_iter=1, random_state=0)

    all_res.append(local_imp)

    perm_imp += np.mean(local_imp, axis=0)

    print('')

    

top_feats = np.argwhere(perm_imp < 0).flatten()

top_feats
def metric(y_true, y_pred):

    metrics = []

    for _target in trainTargetsS.columns:

        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels = [0,1]))

    return np.mean(metrics)
N_STARTS = 7

tf.random.set_seed(42)



res = trainTargetsS.copy()

submission.loc[:, trainTargetsS.columns] = 0

res.loc[:, trainTargetsS.columns] = 0



for seed in range(N_STARTS):

    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=7, random_state=seed, shuffle=True).split(trainTargetsS, trainTargetsS)):

        print(f'Fold {n}')

    

        model = create_model(len(top_feats))

        checkpoint_path = f'repeat:{seed}_Fold:{n}.hdf5'

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

        cb_checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True,

                                     save_weights_only = True, mode = 'min')

        model.fit(train.values[tr][:, top_feats],

                  trainTargetsS.values[tr],

                  validation_data=(train.values[te][:, top_feats], trainTargetsS.values[te]),

                  epochs=35, batch_size=128,

                  callbacks=[reduce_lr_loss, cb_checkpt], verbose=2

                 )

        

        model.load_weights(checkpoint_path)

        test_predict = model.predict(test.values[:, top_feats])

        val_predict = model.predict(train.values[te][:, top_feats])

        

        submission.loc[:, trainTargetsS.columns] += test_predict

        res.loc[te, trainTargetsS.columns] += val_predict

        print('')

    

submission.loc[:, trainTargetsS.columns] /= ((n+1) * N_STARTS)

res.loc[:, trainTargetsS.columns] /= N_STARTS
print(f'OOF Metric: {metric(trainTargetsS, res)}')

submission.loc[test['cp_type'] == 1, trainTargetsS.columns] = 0

submission.to_csv('submission.csv', index = False)

submission.head()