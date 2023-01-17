# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nilearn import plotting, image 

import nibabel as nb

import h5py

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.auto import tqdm
# Nifti MASK image

brain_mask = nb.load('../input/trends-assessment-prediction/fMRI_mask.nii')

plotting.plot_roi(brain_mask, title='fMRI_mask.nii');
from nilearn import datasets

aal = datasets.fetch_atlas_aal();

# This is just supposed to be an example - so it's not too important

try:

    plotting.plot_roi(aal['maps'], title='Example of a Brain Atlas (aal)');

except:

    print("Probably time out")
fnc_10 = next(pd.read_csv('../input/trends-assessment-prediction/fnc.csv', low_memory=True, chunksize=5))

fnc_10.head()
fnc10_cols = fnc_10.columns.to_list()[1:]

fnc10_cols_filtered = [i.split('_')[0] for i in fnc10_cols]

print(np.unique(fnc10_cols_filtered))
# Let's extract the indices for the different networks

# Network index:

ntwk_idx = {}

network_names = np.unique([i[:3] for i in fnc10_cols_filtered])

for ii in network_names:

    ntwk_idx[ii] = np.unique([np.int(i.split('(')[-1].split(')')[0]) for i in fnc10_cols_filtered if ii in i])

    

# Look up matrix index

icn_number = pd.read_csv('../input/trends-assessment-prediction/ICN_numbers.csv')



icn_idx = {}



for jj in ntwk_idx.keys():

    icn_idx[jj] = np.array(icn_number.index[icn_number.ICN_number.isin(ntwk_idx[jj])])
# We load the data using h5py

test_mat1 = h5py.File('../input/trends-assessment-prediction/fMRI_test/11000.mat', mode='r')

print(test_mat1.keys())

test_mat1 = np.array(test_mat1.get('SM_feature'))

print('Dimensions of ICA feature map')

print(test_mat1.shape)

print('Dimenions of the brain mask')

print(brain_mask.shape)



## Let's also load a second participant

test_mat2 = h5py.File('../input/trends-assessment-prediction/fMRI_test/10006.mat', mode='r')

test_mat2 = np.array(test_mat2.get('SM_feature'))
# Somehow nilearn is not happy with plotting matrices anymore - so we have to create a nifti first:

def map_for_plotting(mat, brain_mask):

    # Assuming that we provide a 3D image

    # image.new_img_like creates a nifti by applying informaiton from the soure image (here brain_mask),

    # like the affine to a matrix.

    return image.new_img_like(brain_mask, mat.transpose([2, 1, 0]))
# Let's extract the indices for the different average networks

sample_maps1 = {}

sample_maps2 = {}

for ii in icn_idx.keys():

    # indices -1 because matlab

    sample_maps1[ii] = map_for_plotting(test_mat1[icn_idx[ii] -1].mean(0), brain_mask)

    sample_maps2[ii] = map_for_plotting(test_mat2[icn_idx[ii] -1].mean(0), brain_mask)
fig, axes = plt.subplots(len(sample_maps1), 2, figsize=(20, 10))



for n, ii in enumerate(sample_maps1.keys()):

    # We are plotting glass brains here - a nice way to visualize brain maps

    plotting.plot_glass_brain(sample_maps1[ii], title=ii, axes=axes[n, 0], plot_abs=False)

    plotting.plot_glass_brain(sample_maps2[ii], title=ii, axes=axes[n, 1], plot_abs=False)

axes[0, 0].set_title('Networks for Participant 1');

axes[0, 1].set_title('Networks for Participant 2');
# This is probably totally inefficient - but let's try it

icn_mat_idx = icn_number.T.to_dict('list')

# Reverse the matrix:

icn_mat_idx = {i[0]: j for i, j in zip(icn_mat_idx.values(), icn_mat_idx.keys())}

# Map names to indices

name_matrix = {}



for fnco in fnc10_cols:

    name_matrix[fnco] = ([np.int(icn_mat_idx[np.int(i.split(')')[0])]) for i in fnco.split('(')[1:]])

    

# And now create a sample connectivity matrix:

con_matrix1 = np.zeros((53, 53))

con_matrix2 = np.zeros((53, 53))



for n in fnc10_cols:

    r_, c_ = name_matrix[n]

    con_matrix1[c_, r_] = fnc_10.iloc[0, :][n]

    con_matrix2[c_, r_] = fnc_10.iloc[1, :][n]

# And now add the transpose - its symmetrix

con_matrix1 += con_matrix1.T

con_matrix2 += con_matrix2.T



# Prepare labeling:

col_halves = np.array([jj.split('_')[-1]  for jj in name_matrix.keys()])

_, idx = np.unique(col_halves, return_index=True)

col_labels = col_halves[np.sort(idx)]
fig, ax = plt.subplots(1, 2,figsize=(20, 7.5))



sns.heatmap(con_matrix1, cmap='coolwarm', square=True, ax=ax[0], 

            xticklabels=col_labels, 

            yticklabels=col_labels, cbar=False, center=0, vmin=-1, vmax=1)



sns.heatmap(con_matrix2, cmap='coolwarm', square=True, ax=ax[1], 

            xticklabels=col_labels, 

            yticklabels=col_labels, cbar=False, center=0, vmin=-1, vmax=1)



ax[0].set_title('Example 1')

ax[1].set_title('Example 2');
plotting.plot_anat(datasets.load_mni152_template(), title='MNI template');
scores = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')

scores = scores.set_index('Id')

scores.head()
scores.isna().sum(0)
scores = scores.dropna()
fig, axes = plt.subplots(2, 3, figsize=(15, 5))



for ax, data in zip(axes.flatten()[:5], scores.columns[:5]):

    deciles = np.percentile(scores[data].values, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    

    sns.distplot(scores[data], ax=ax)

    # Ugly, but whatever

    for de in deciles:

        ax.axvline(de, c='black')

    

    ax.set_title(data)

    

axes.flatten()[-1].set_axis_off()



plt.tight_layout()
sns.heatmap(scores.corr())
# Add discretization to data:

for sc in scores.columns[:5]:

    deciles = np.percentile(scores.loc[:, sc], [20, 40,  60, 80])

    discr = np.digitize(scores.loc[:, sc], deciles)

    scores.loc[:, sc + '_discrete'] = discr.astype(str)

    

# Everything to one variable:

scores.loc[:, 'stratify'] = (scores['age_discrete'] + '_'

                             + scores['domain1_var1_discrete'] + '_' 

                             + scores['domain2_var2_discrete'])
scores.stratify.value_counts()
# And now draw a stratified sample, we will statistically analyse 20% of the data

from sklearn.model_selection import train_test_split



train_idx, _ = train_test_split(scores.index, train_size=0.2, random_state=223, stratify=scores.stratify)
scores_stat = scores.loc[train_idx]



fig, axes = plt.subplots(2, 3, figsize=(15, 5))



for ax, data in zip(axes.flatten()[:5], scores_stat.columns[:5]):

    deciles = np.percentile(scores_stat[data].values, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    

    sns.distplot(scores_stat[data], ax=ax)

    sns.distplot(scores[data], ax=ax)

    ax.legend(['Subsample', 'Original'])

    # Ugly, but whatever

    for de in deciles:

        ax.axvline(de, c='black')

    

    ax.set_title(data)



    

axes.flatten()[-1].set_axis_off()



plt.tight_layout()
fnc = pd.read_csv('../input/trends-assessment-prediction/fnc.csv', index_col='Id')
fnc_sample = fnc.loc[train_idx, :]
# Correlations: 

corr_df = []

for col_score in ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']:

    tmp_corr = fnc_sample.corrwith(scores_stat.loc[:, col_score]).to_frame().transpose()

    corr_df.append(tmp_corr)



fig, axes = plt.subplots(1, 5, figsize=(15, 5), sharex=True, sharey=True)



for col_score, ax, tmp_df in zip(['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2'], axes.flatten(), corr_df):

    ax.hist(tmp_df.transpose().values.ravel())

    ax.set_title(col_score)

plt.suptitle('Histograms of pearson correlations');
fig, axes = plt.subplots(2, 3, figsize=(15,10), sharex=True, sharey=True)



for col_score, ax, tmp_corr in zip(['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2'], axes.flatten(), corr_df):



    perc_ = np.percentile(np.abs(tmp_corr.values.ravel()), 90)

    tmp_matrix = np.zeros((53, 53))

    for n in fnc10_cols:

        r_, c_ = name_matrix[n]

        tmp_matrix[c_, r_] = tmp_corr.iloc[0, :][n]

            

    tmp_matrix[np.abs(tmp_matrix) < perc_] = 0

    tmp_matrix += tmp_matrix.T



    sns.heatmap(tmp_matrix, cmap='coolwarm', square=True, ax=ax, 

            xticklabels=col_labels, 

            yticklabels=col_labels, cbar=False, center=0, vmin=-0.25, vmax=0.25)



    ax.set_title(col_score)

    

axes.flatten()[-1].set_axis_off()

print(perc_)

plt.tight_layout()
from sklearn.decomposition import PCA



pca = PCA(n_components=0.8, whiten=True, svd_solver='full')

pca.fit(fnc_sample.values)



components = pca.transform(fnc_sample.values)

components = pd.DataFrame(components, index=scores_stat.index)
fig, axes = plt.subplots(1,3, figsize=(15,5))



axes[0].plot(pca.explained_variance_ratio_[:21])

axes[0].set(title='Elbowplot of PCA components', ylabel='Explained Variance', xlabel='Components')



tmp_matrix1 = np.zeros((53, 53))

tmp_matrix2 = np.zeros((53, 53))

for i, n in enumerate(fnc10_cols):

    r_, c_ = name_matrix[n]

    tmp_matrix1[c_, r_] = pca.components_[0, i]

    tmp_matrix2[c_, r_] = pca.components_[1, i]



tmp_matrix1 += tmp_matrix1.T

tmp_matrix2 += tmp_matrix2.T



sns.heatmap(tmp_matrix1, cmap='coolwarm', square=True, ax=axes[1], 

        xticklabels=col_labels, 

        yticklabels=col_labels, cbar=False, center=0)

axes[1].set(title='Component 0')

axes[1].set_xticklabels(axes[1].get_xmajorticklabels(),  fontsize=6)



sns.heatmap(tmp_matrix2, cmap='coolwarm', square=True, ax=axes[2], 

        xticklabels=col_labels, 

        yticklabels=col_labels, cbar=False, center=0)

axes[2].set(title='Component 1')

axes[2].set_xticklabels(axes[2].get_xmajorticklabels(), fontsize=6)



plt.tight_layout()
pca_corr = []

for kk in range(10):

    pca_corr.append(scores_stat[['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']].corrwith(components.loc[:,kk]))



pd.concat(pca_corr, axis=1)
loadings_samp = pd.read_csv('../input/trends-assessment-prediction/loading.csv', index_col='Id')
loadings_samp = loadings_samp.loc[train_idx, :]
# Correlations: 

fig, axes = plt.subplots(1, 5, figsize=(15,5))

load_corr = []

for ax, col_score in zip(axes.flatten(), ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']):

    tmp_corr = loadings_samp.corrwith(scores_stat.loc[:, col_score]).to_frame().transpose()

    load_corr.append(tmp_corr)

    ax.hist(tmp_corr.values.ravel())

    ax.set_title(col_score)

    

load_corr = pd.concat(load_corr)

load_corr.loc[:, 'Assessment'] = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2'] 

load_corr.set_index('Assessment', inplace=True)
try:

    basc_data = datasets.fetch_atlas_basc_multiscale_2015(version='sym', data_dir=None, resume=True, verbose=1)

except:

    print("Probably time out")
basc_197 = nb.load(basc_data['scale197'])

plotting.plot_roi(basc_197)
from nilearn import input_data

# We also use the brain_mask from the beginning

basc197_masker = input_data.NiftiLabelsMasker(basc_197, mask_img=brain_mask)



def load_matlab(participant_id, masker, path='../input/trends-assessment-prediction/fMRI_train/'):

    mat = np.array(h5py.File(f'{path}{participant_id}.mat', mode='r').get('SM_feature'))

    mat = masker.fit_transform(nb.Nifti1Image(mat.transpose([3,2,1,0]), affine=masker.mask_img.affine))

    return mat.flatten()
# This takes ages about (like 8 min ... so time 5 for the whole data set)

from joblib import Parallel, delayed



sm_data = Parallel(n_jobs=-1)(delayed(load_matlab)(ii, basc197_masker) for ii in tqdm(list(train_idx)))

sm_data = np.stack(sm_data)
from sklearn.decomposition import FastICA

pca_2 = PCA(n_components=0.6, whiten=True)

pca_2.fit(sm_data)



components2 = pca_2.fit_transform(sm_data)
fig, axes = plt.subplots(1,1, figsize=(15,5))



axes.plot(pca_2.explained_variance_ratio_[:30])

axes.set(title='Elbowplot of PCA components', ylabel='Explained Variance', xlabel='Components')



plt.tight_layout()
components2 = pd.DataFrame(components2, index=scores_stat.index)

pca2_corr = []

for kk in range(20):

    pca2_corr.append(scores_stat[['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']].corrwith(components2.loc[:,kk]))



pca2_scorr = pd.concat(pca2_corr, axis=1)

pca2_scorr
fig, axes = plt.subplots(1, 5, figsize=(15,5), sharex=True, sharey=True)



for n, ax in enumerate(axes.flatten()):

    ax.plot(pca2_scorr.iloc[n, :])

    ax.set_title(pca2_scorr.index[n])

    ax.axhline(0)
# Not the best approach but let's load some data again and delete some

import gc

try:

    del fnc

    del sm_data

    del pca

    del pca_2

except:

    pass

gc.collect()
# Loading data again - we have the scores with the stratifier variable alread

fnc = pd.read_csv('../input/trends-assessment-prediction/fnc.csv', index_col='Id')

loading = pd.read_csv('../input/trends-assessment-prediction/loading.csv', index_col='Id')
from sklearn.linear_model import RidgeCV

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA
fnc_train = fnc.loc[scores.index, :]

loading_train = loading.loc[scores.index, :]
# Test whether indices align

assert np.all(fnc_train.index == loading_train.index) 

assert np.all(fnc_train.index == scores.index) 

assert np.all(loading_train.index == scores.index)
SKF = StratifiedKFold(n_splits=4)

targets = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

weighting = [.3, .175, .175, .175, .175]
# I hope I did the calculation correctly

def absolute_normalized_error(y_true, y_pred, multioutput):

    output_errors = np.sum(np.abs(y_pred - y_true), axis=0) / np.sum(y_pred, axis=0)    

    return np.average(output_errors, weights=multioutput)

REG_FNC = make_pipeline(PCA(n_components=50, whiten=False), RobustScaler(), RidgeCV(alphas=np.logspace(-5, 5, 11))) # Some dimensionality reduction might be in order

REG_LOA = make_pipeline(RobustScaler(), RidgeCV(alphas=np.logspace(-5, 5, 11))) # Not so much here



trues, preds_fnc, preds_load, preds_comb = [], [], [], []

scores_fnc, scores_load, scores_comb = [], [], []



for tr, te in SKF.split(fnc_train, scores.stratify):

    REG_FNC.fit(fnc_train.iloc[tr, :].values, scores.iloc[tr][targets])

    REG_LOA.fit(loading_train.iloc[tr, :].values, scores.iloc[tr][targets])

    

    preds_fnc.append(REG_FNC.predict(fnc_train.iloc[te,:]))

    preds_load.append(REG_LOA.predict(loading_train.iloc[te,:]))

    preds_comb.append((preds_fnc[-1] + preds_load[-1]) / 2)

    trues.append(scores.iloc[te][targets])

    scores_fnc.append(absolute_normalized_error(trues[-1], preds_fnc[-1],  multioutput=weighting))

    scores_load.append(absolute_normalized_error(trues[-1], preds_load[-1],  multioutput=weighting))

    scores_comb.append(absolute_normalized_error(trues[-1], preds_comb[-1],  multioutput=weighting))

    
print(f'Error based on FNC: {np.mean(scores_fnc)} +/- {np.std(scores_fnc)}')

print(f'Error based on Load: {np.mean(scores_load)} +/- {np.std(scores_load)}')

print(f'Error based on Load: {np.mean(scores_comb)} +/- {np.std(scores_comb)}')
REG_FNC.fit(fnc_train, scores[targets])

REG_LOA.fit(loading_train, scores[targets])
# Get the test data

sample_submission = pd.read_csv('../input/trends-assessment-prediction/sample_submission.csv')
# Extract the test index

test_index = sample_submission.Id.str.split('_', expand=True)[0].unique().astype('int')

fnc_test = fnc.loc[test_index, :]

loading_test = loading.loc[test_index, :]
# Calculate the average prediction value

prediction = (REG_FNC.predict(fnc_test) + REG_LOA.predict(loading_test)) / 2
# Submit prediction

predictions = pd.DataFrame(prediction, index=test_index, columns=targets).reset_index()

predictions = predictions.rename(columns={'index': 'Id'})

predictions = predictions.melt(id_vars='Id', value_vars=targets, value_name='Predicted')

predictions.loc[:, 'Id'] = predictions.loc[:, 'Id'].astype(str) + '_' + predictions.loc[:, 'variable']

predictions = predictions[['Id', 'Predicted']]

predictions.to_csv('ridge_baseline_submission.csv', index=False)