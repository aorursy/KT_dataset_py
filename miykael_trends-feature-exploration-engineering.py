import os
import numpy as np
import pandas as pd
from glob import glob
from os.path import join as opj

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from tqdm.notebook import tqdm
path = '/kaggle/input/trends-assessment-prediction/'
# Load targets
targets = pd.read_csv(opj(path, 'train_scores.csv')).set_index('Id')
# Let's also create the rotated domain2 targets
rot = 0.90771256655

def rotate_origin(x, y, radians):
    """Rotate a point around the origin (0, 0)."""
    xx = x * np.cos(radians) + y * np.sin(radians)
    yy = -x * np.sin(radians) + y * np.cos(radians)
    return np.array([xx, yy]).T

d2 = rotate_origin(targets.iloc[:, 3].values, targets.iloc[:, 4].values, rot)
targets['d21_rot'] = d2[:, 0]
targets['d22_rot'] = d2[:, 1]
# Let's apply the power transformation to make the value distribution gaussian
pow_age = 1.0
pow_d1v1 = 1.5
pow_d1v2 = 1.5
pow_d2v1 = 1.5
pow_d2v2 = 1.5
pow_d21 = 1.5
pow_d22 = 1
powers = [pow_age, pow_d1v1, pow_d1v2, pow_d2v1, pow_d2v2, pow_d21, pow_d22 ]

for i, col in enumerate(targets.columns):
    targets[col] = np.power(targets[col], powers[i])
from sklearn.preprocessing import StandardScaler

# And last but not least, let's scale the target features using ab
scaler = StandardScaler()
targets.iloc[:, :] = scaler.fit_transform(targets)
targets.head()
# Extract ID to separate train and test set
train_id = targets.index.values
sample_submission = pd.read_csv(opj(path, 'sample_submission.csv'))
test_id = np.unique(sample_submission.Id.str.split('_', expand=True)[0].astype('int'))
print(train_id.shape, test_id.shape)
# Load ICs from the loading file and separate them into train and test set
df_ic = pd.read_csv(opj(path, 'loading.csv'))
ic_train = df_ic[df_ic.Id.isin(train_id)].set_index('Id')
ic_test = df_ic[df_ic.Id.isin(test_id)].set_index('Id')
print(ic_train.shape, ic_test.shape)
# Load FNCs from file and separate them into train and test set
df_fnc = pd.read_csv(opj(path, 'fnc.csv'))
fnc_train = df_fnc[df_fnc.Id.isin(train_id)].set_index('Id')
fnc_test = df_fnc[df_fnc.Id.isin(test_id)].set_index('Id')
print(fnc_train.shape, fnc_test.shape)
def plot_corr_matrix(df_train, df_test, c_restrict=200):

    # Correlation matrix for ICA components
    fig, ax = plt.subplots(ncols=3, figsize=(20, 10))
    abs_max = 1.0
    sns.heatmap(df_train.iloc[:, :c_restrict].corr(), square=True, vmin=-abs_max, vmax=abs_max, cbar=False, ax=ax[0]);
    sns.heatmap(df_test.iloc[:, :c_restrict].corr(), square=True, vmin=-abs_max, vmax=abs_max, cbar=False, ax=ax[1]);
    sns.heatmap(df_train.iloc[:, :c_restrict].corr()-df_test.iloc[:, :c_restrict].corr(),
                square=True, vmin=-0.33, vmax=0.33, cbar=False, ax=ax[2]);
    ax[0].set_title('Train')
    ax[1].set_title('Test')
    ax[2].set_title('Difference (Train - Test)');
# Correlation matrix for IC features
plot_corr_matrix(ic_train, ic_test, c_restrict=100)
# Correlation matrix for FNC features
plot_corr_matrix(fnc_train, fnc_test, c_restrict=100)
def plot_corr_matrix_target(targets, df_train, c_restrict=100):

    # Merge target and feature matrix
    df_temp = pd.merge(targets.reset_index(), df_train.reset_index())
    df_temp = df_temp.set_index('Id').iloc[:, :c_restrict]
    
    # Correlation matrix for ICA components
    plt.figure(figsize=(16, 3))
    sns.heatmap(df_temp.corr().iloc[:7, 7:], square=True,
                vmin=-0.5, vmax=0.5, cbar=False, cmap='Spectral');
# Correlation between IC features and targets
plot_corr_matrix_target(targets, ic_train, c_restrict=100)
# Correlation between FNC features and targets
plot_corr_matrix_target(targets, fnc_train, c_restrict=100)
# Show highest correlation with target variables and IC dataset
df_corr = pd.concat([np.abs(ic_train.corrwith(targets.iloc[:, i])).sort_values(ascending=False).reset_index(drop=True) for i in range(7)], axis=1)
df_corr.columns = targets.columns
df_corr.head(5)
# Show highest correlation with target variables and FNC dataset
df_corr = pd.concat([np.abs(fnc_train.corrwith(targets.iloc[:, i])).sort_values(ascending=False).reset_index(drop=True) for i in range(7)], axis=1)
df_corr.columns = targets.columns
df_corr.head(5)
def plot_rotation_correlations(df_data, targets, ttt=3):

    corr_max = []
    for r in np.linspace(0, 3.14, 100):

        bla = targets.iloc[:, i].copy()
        bla.iloc[:] = rotate_origin(targets.iloc[:, ttt].values, targets.iloc[:, ttt+1].values, r)[:, 0]
        corr_max.append([np.rad2deg(r), df_data.corrwith(bla).sort_values(ascending=False).reset_index(drop=True).abs().max()])

    corr_max1 = np.array(corr_max)
    plt.figure(figsize=(14, 4))
    plt.scatter(corr_max1[:, 0], corr_max1[:, 1], s=3);

    corr_max = []
    for r in np.linspace(0, 3.14, 100):

        bla = targets.iloc[:, i].copy()
        bla.iloc[:] = rotate_origin(targets.iloc[:, ttt].values, targets.iloc[:, ttt+1].values, r)[:, 1]
        corr_max.append([np.rad2deg(r), df_data.corrwith(bla).sort_values(ascending=False).reset_index(drop=True).abs().max()])

    corr_max2 = np.array(corr_max)
    plt.scatter(corr_max2[:, 0], corr_max2[:, 1], s=3);

    best_corr = corr_max1[np.argmin(np.abs(corr_max1[:, 1] - corr_max2[:, 1])), 1]
    best_rot = corr_max1[np.argmin(np.abs(corr_max1[:, 1] - corr_max2[:, 1])), 0]
    plt.title('Equal correlation of %.4f\nat rotation of %.4f radians' % (best_corr, best_rot))
    plt.legend(['domain2_var1_rot', 'domain2_var2_rot'])
plot_rotation_correlations(ic_train, targets, ttt=3)
plot_rotation_correlations(fnc_train, targets, ttt=3)
# Number of columns to investigate
n_invest = 10
sns.pairplot(ic_train.iloc[:, :n_invest], diag_kind="kde", corner=True);
sns.pairplot(fnc_train.iloc[:, :n_invest], diag_kind="kde", corner=True);
def plot_markers(key, df_temp, ncolmarker=5, split_at=5, plot_max=15):

    # Restrict dataframe to first X features
    df_temp = df_temp.iloc[:, :plot_max]

    # Compute dataset selecters
    ncolumns = np.arange(df_temp.shape[1])
    selecter = np.split(ncolumns, ncolumns[::split_at][1:])

    for s in selecter:

        print(key, s)
        df_temp.iloc[:, s].plot(kind='line',subplots=True, sharex=True, marker='.', lw=0,
                                ms=10, markeredgecolor='k', markeredgewidth=0.3,
                     figsize=(5 * ncolmarker, 4 * df_temp.iloc[:, s].shape[1]//ncolmarker), layout=(-1,ncolmarker));
        plt.show()
plot_markers('Visualization of IC features:', ic_train)
plot_markers('Visualization of fNC features:', fnc_train)
# To bypass feature extraction and load precomputed files
load_pre_computed_files = True
import h5py
import nilearn as nl
from nilearn import image, plotting
# Load brain mask
mask = nl.image.load_img(opj(path, 'fMRI_mask.nii'))
# This function was inspired by a fellow kaggler, who I can't find the source anymore
def read_img(filename, mask):
    with h5py.File(filename, 'r') as f:
        data = np.array(f['SM_feature'], dtype='float32')

    # It's necessary to reorient the axes, since h5py flips axis order
    data = np.moveaxis(data, [0, 1, 2, 3],
                             [3, 2, 1, 0])

    img = nl.image.new_img_like(mask, data, affine=mask.affine, copy_header=True)
    return img
# Only convert every n-th subject
sub_sample = 100
# Rewrite mat file to compressed NIfTI
directory='fMRI_train'
if not os.path.exists(directory):
    os.makedirs(directory)
for fname in tqdm(sorted(glob(opj(path, directory, '*.mat')))[::sub_sample]):
    new_filename = fname.replace('.mat', '.nii.gz')
    new_filename = new_filename.replace('/kaggle/input/trends-assessment-prediction/', '')
    read_img(fname, mask).to_filename(new_filename)
# Rewrite mat file to compressed NIfTI
directory='fMRI_test'
if not os.path.exists(directory):
    os.makedirs(directory)
for fname in tqdm(sorted(glob(opj(path, directory, '*.mat')))[::sub_sample]):
    new_filename = fname.replace('.mat', '.nii.gz')
    new_filename = new_filename.replace('/kaggle/input/trends-assessment-prediction/', '')
    read_img(fname, mask).to_filename(new_filename)
# Load data from one subject
img = image.load_img(sorted(glob('fMRI_train/*.nii.gz'))[0])

# Mask the image to only look at correlation within voxels which have a value
data = img.get_fdata()[mask.get_fdata()>0]

# Compute correlation matrix
corr_matrix = np.corrcoef(data.T)
plt.figure(figsize=(6, 6))
sns.heatmap(corr_matrix, square=True, cbar=True);
# Only keep upper triangular correlation matrix without diagonal
triangular_mask = np.ravel(np.triu(np.ones((53, 53)), k=1))>0.5
corr_values = np.ravel(corr_matrix)[triangular_mask]
print(corr_values.shape, corr_values)
# Let's create an output folder to store the new features
directory='datasets'
if not os.path.exists(directory):
    os.makedirs(directory)
if load_pre_computed_files:

    # Load precomputed intra subject correlation data for the training set
    hdf_path = opj('/kaggle', 'input', 'corr-features', 'intra_corr_train.h5')
    df_corr_intra_train = pd.read_hdf(hdf_path)

else:

    # Collect results
    corr_results = {}

    # Collect all train files
    train_files = sorted(glob('fMRI_train/*.nii.gz'))

    for t in tqdm(train_files):

        try:
            # Load mean image
            img = image.load_img(t)
            data = img.get_fdata()[mask.get_fdata()>0]

            t_id = t.split('/')[1].split('.')[0]
            corr_results[t_id] = np.ravel(np.corrcoef(data.T))

        except:
                print("Wasn't able to load: ", t)

    df_corr = pd.DataFrame(corr_results).T
    df_corr.columns = ['c%02d_c%02d' % (i + 1, j + 1)
                       for i in range(53) for j in range(53)]

    # Only keep upper triangular correlation matrix without diagonal
    triangular_mask = np.ravel(np.triu(np.ones((53, 53)), k=1))>0.5
    df_corr_intra_train = df_corr.loc[:, triangular_mask]

    # Save everything in CSV file
    df_corr_intra_train.to_hdf('datasets/df_corr_intra_train.hdf5', key='df_corr_intra_train', mode='w')

# Plopt head of dataframe
df_corr_intra_train.head()
if load_pre_computed_files:

    # Load precomputed intra subject correlation data for the training set
    hdf_path = opj('/kaggle', 'input', 'corr-features', 'intra_corr_test.h5')
    df_corr_intra_test = pd.read_hdf(hdf_path)

else:

    # Collect results
    corr_results = {}

    # Collect all test files
    test_files = sorted(glob('fMRI_test/*.nii.gz'))

    for t in tqdm(test_files):

        try:
            # Load mean image
            img = image.load_img(t)
            data = img.get_fdata()[mask.get_fdata()>0]

            t_id = t.split('/')[1].split('.')[0]
            corr_results[t_id] = np.ravel(np.corrcoef(data.T))

        except:
                print(t)

    df_corr = pd.DataFrame(corr_results).T
    df_corr.columns = ['c%02d_c%02d' % (i + 1, j + 1)
                       for i in range(53) for j in range(53)]

    # Only keep upper triangular correlation matrix without diagonal
    triangular_mask = np.ravel(np.triu(np.ones((53, 53)), k=1))>0.5
    df_corr_intra_test = df_corr.loc[:, triangular_mask]

    # Save everything in CSV file
    df_corr_intra_test.to_hdf('datasets/intra_corr_test.hdf5', key='intra_corr_test', mode='w')

# Plopt head of dataframe
df_corr_intra_test.head()
# Correlation matrix for IC features
plot_corr_matrix(df_corr_intra_train, df_corr_intra_test, c_restrict=100)
# Correlation between IC features and targets
plot_corr_matrix_target(targets, df_corr_intra_train, c_restrict=100)
# Show highest correlation with target variables and IC dataset
df_corr = pd.concat([np.abs(df_corr_intra_train.corrwith(targets.iloc[:, i])).sort_values(ascending=False).reset_index(drop=True) for i in range(7)], axis=1)
df_corr.columns = targets.columns
df_corr.head(5)
# Pairplots between intra correlation values and targets
sns.pairplot(df_corr_intra_train.iloc[:, :n_invest], diag_kind="kde", corner=True);
# Visualization of values in dataset
plot_markers('Visualization of intra correlation features:', df_corr_intra_train)
from nilearn import image, plotting, masking
from nilearn.regions import connected_regions
# Creates the mean image for a given component
def get_mean_component(filenames, comp_ID=0):
    mean = image.math_img('img * 0', img=mask)
    for f in filenames:
        img = image.load_img(f).slicer[..., comp_ID]
        mean = image.math_img('mean + img', mean=mean, img=img)
    mean = image.math_img('img / %f' % len(filenames), img=mean)
    return mean
# Creating an output folder to store the average maps
directory='fMRI_maps'
if not os.path.exists(directory):
    os.makedirs(directory)
# Extract the mean images
n_maps = 8    # Change this parameter to 53 to get all components

filenames = sorted(glob('fMRI_train/*.nii.gz'))
for idx in tqdm(range(n_maps)):
    mean = get_mean_component(filenames, comp_ID=idx)
    mean.to_filename('fMRI_maps/mean_%02d.nii.gz' % (idx + 1))
# Let's plot the first n-th average maps (threshold at 95% max value)
for idx in range(n_maps):
    img = image.load_img('fMRI_maps/mean_%02d.nii.gz' % (idx + 1))
    data = img.get_fdata()
    threshold = np.percentile(data[data!=0], 95)
    img_thr = image.threshold_img(img, threshold=threshold)
    img_regions = image.mean_img(connected_regions(img_thr, min_region_size=4000)[0])
    plotting.plot_glass_brain(img_regions, black_bg=True, display_mode='lyrz',
                              title='mean_%02d' % (idx + 1))
    plt.show()
def combine_brain_values(didx='train'):
    """Helper function to combine all 53 component CSV files into one big one"""

    # List of file names
    csv_files = sorted(glob('datasets/inter_corr_*_%s_*.csv' % didx))

    # Create empty ID list
    merger = pd.read_csv(csv_files[0]).set_index('Id')
    merger.columns = [c + '_%02d' % 1 for c in merger.columns]

    # Go through files and concatenate them
    for i, f in enumerate(csv_files[1:]):

        new_df = pd.read_csv(f).set_index('Id')
        new_df.columns = [c + '_%02d' % (i + 2) for c in new_df.columns]

        merger = pd.merge(merger, new_df, on='Id')

    return merger

if load_pre_computed_files:

    # Load precomputed inter subject correlation data for the training set
    hdf_path = opj('/kaggle', 'input', 'corr-features', 'inter_corr_train.h5')
    df_corr_inter_train = pd.read_hdf(hdf_path)

else:

    # Collect value metrics from images
    train_files = sorted(glob('fMRI_train/*.nii.gz'))

    for idx in tqdm(range(n_maps)):

        # Load mean image
        mean = image.load_img('fMRI_maps/mean_%02d.nii.gz' % (idx + 1))
        data_mean = mean.get_fdata()[mask.get_fdata()>0]

        # Compute binary mask for region
        mask_region = data_mean > np.percentile(data_mean, 99)

        # Store results in results file
        results = {}

        for t in train_files:

            try:
                # Get file name
                t_id = t.split('/')[1].split('.')[0]

                # Load current volume
                img = image.index_img(t, idx)

                # Only extract data values from within mask
                data_img = img.get_fdata()[mask.get_fdata()>0]

                # Collect correlation coefficient to mean image
                corr_coef = np.corrcoef(data_img, data_mean)[0, 1]

                results[t_id] = [t_id, corr_coef]
            except:
                print(t)

        # Store result in CSV file
        df_results = pd.DataFrame(results).T
        df_results.columns = ['Id', 'corr_coef']
        df_results.to_csv('datasets/inter_corr_train_%02d.csv' % (idx + 1), index=False)

    # Load brain value components
    df_corr_inter_train = combine_brain_values(didx='train')
        
df_corr_inter_train.head()
if load_pre_computed_files:

    # Load precomputed inter subject correlation data for the training set
    hdf_path = opj('/kaggle', 'input', 'corr-features', 'inter_corr_test.h5')
    df_corr_inter_test = pd.read_hdf(hdf_path)

else:

    # Collect results
    test_files = sorted(glob('fMRI_test/*.nii.gz'))

    for idx in tqdm(range(n_maps)):

        # Load mean image
        mean = image.load_img('fMRI_maps/mean_%02d.nii.gz' % (idx + 1))
        data_mean = mean.get_fdata()[mask.get_fdata()>0]

        # Compute binary mask for region
        mask_region = data_mean > np.percentile(data_mean, 99)

        # Store results in results file
        results = {}

        for t in test_files:

            try:
                # Get file name
                t_id = t.split('/')[1].split('.')[0]

                # Load current volume
                img = image.index_img(t, idx)

                # Only extract data values from within mask
                data_img = img.get_fdata()[mask.get_fdata()>0]

                # Collect correlation coefficient to mean image
                corr_coef = np.corrcoef(data_img, data_mean)[0, 1]

                results[t_id] = [t_id, corr_coef]
            except:
                print(t)

        # Store result in CSV file
        df_results = pd.DataFrame(results).T
        df_results.columns = ['Id', 'corr_coef']
        df_results.to_csv('datasets/inter_corr_test_%02d.csv' % (idx + 1), index=False)
        
    # Load brain value components
    df_corr_inter_test = combine_brain_values(didx='test')
        
df_corr_inter_test.head()
# Correlation matrix for IC features
plot_corr_matrix(df_corr_inter_train, df_corr_inter_test, c_restrict=100)
# Correlation between IC features and targets
plot_corr_matrix_target(targets, df_corr_inter_train, c_restrict=100)
# Show highest correlation with target variables and IC dataset
df_corr = pd.concat([np.abs(df_corr_inter_train.corrwith(targets.iloc[:, i])).sort_values(ascending=False).reset_index(drop=True) for i in range(7)], axis=1)
df_corr.columns = targets.columns
df_corr.head(5)
# Pairplots between intra correlation values and targets
sns.pairplot(df_corr_inter_train.iloc[:, :n_invest], diag_kind="kde", corner=True);
# Visualization of values in dataset
plot_markers('Visualization of inter correlation features:', df_corr_intra_train)
# Create merged dataset
merge_train = pd.merge(ic_train.reset_index(), fnc_train.reset_index()).set_index('Id')
merge_train = pd.merge(merge_train, df_corr_intra_train, left_index=True, right_index=True)
merge_train = pd.merge(merge_train, df_corr_inter_train, left_index=True, right_index=True)
print(merge_train.shape)

merge_test = pd.merge(ic_test.reset_index(), fnc_test.reset_index()).set_index('Id')
merge_test = pd.merge(merge_test, df_corr_intra_test, left_index=True, right_index=True)
merge_test = pd.merge(merge_test, df_corr_inter_test, left_index=True, right_index=True)
print(merge_test.shape)
# Detect very frequent extrem values with z-score outliers
df_zscore = (merge_train - merge_train.mean())/merge_train.std()

extrem_ids = []
for above_std, how_many_times in [[4, 8], [5, 4], [6, 2]]:

    # Detect extrem values
    extrem_values = np.sum(df_zscore.abs()>=above_std, axis=1)>=how_many_times
    new_extrems = list(np.array(merge_train[extrem_values].index))
    extrem_ids.extend(new_extrems)
    print('Found %d outliers with an absolute z-score above %d, at least %d times.' % (len(new_extrems), above_std, how_many_times))

extrem_ids = np.unique(extrem_ids)
print('Total of unique outliers found: %d' % len(extrem_ids))
extrem_ids
# Missing values discovered within the features.
outliers = np.ravel([np.array([t for t in train_id if not np.isin(t, ic_train.index.values)])])
outliers = np.hstack((outliers, np.array([t for t in train_id if not np.isin(t, fnc_train.index.values)])))
outliers = np.hstack((outliers, np.array([t for t in train_id if not np.isin(t, df_corr_intra_train.index.values)])))
outliers = np.hstack((outliers, np.array([t for t in train_id if not np.isin(t, df_corr_inter_train.index.values)])))
outliers = np.hstack((outliers, extrem_ids))
outliers = np.unique(outliers)
print(len(outliers))
outliers
# Remove outliers from features
ic_train = ic_train.drop(outliers, errors='ignore')
fnc_train = fnc_train.drop(outliers, errors='ignore')
df_corr_intra_train = df_corr_intra_train.drop(outliers, errors='ignore')
df_corr_inter_train = df_corr_inter_train.drop(outliers, errors='ignore')
merge_train = merge_train.drop(outliers, errors='ignore')
print(ic_train.shape, fnc_train.shape, df_corr_intra_train.shape, df_corr_inter_train.shape, merge_train.shape)
# Remove outliers from target
targets = targets.drop(outliers, errors='ignore')
targets.shape
# Store datasets as hdf5 files
merge_train.to_hdf('datasets/merge_train.h5', key='merge_train', mode='w')
merge_test.to_hdf('datasets/merge_test.h5', key='merge_test', mode='w')
targets.to_hdf('datasets/targets.h5', key='targets', mode='w')
# Store scaler in a pickle file
import joblib
joblib.dump(scaler, 'datasets/targets_scaler.pkl');
# Before quitting, be conscious about space and let's clean our working directory
!rm -rf fMRI* datasets/intra* datasets/inter*
from sklearn.decomposition import PCA
for key, df_temp in [['ic features', ic_train],
                     ['fnc features', fnc_train],
                     ['intra corr features', df_corr_intra_train],
                     ['inter corr features', df_corr_inter_train],
                    ]:

    # Explore explained variance on PCA components
    s = StandardScaler()
    X_scaled = s.fit_transform(df_temp)

    # Create PCA reduced dataset
    pca = PCA()
    pca_train = pca.fit_transform(X_scaled)

    # Explore PCA components
    pve_cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 5))
    plt.title('PCA Explained Variance Ratio: %s' % key)
    plt.step(range(len(pve_cumsum)), pve_cumsum)
    plt.show();
    
    for thresh in [0.8, 0.9, 0.95, 0.99]:
        txt = 'Explained Variance for {}: {}% | Components: {}'.format(
            key, int(thresh * 100),
            np.argmax(pve_cumsum>=thresh))
        print(txt)
import umap
for key, df_temp in [['ic features', ic_train],
                     ['fnc features', fnc_train],
                     ['intra corr features', df_corr_intra_train],
                     ['inter corr features', df_corr_inter_train],
                     ['merged features', merge_train],
                    ]:

    # Explore explained variance on PCA components
    s = StandardScaler()
    X_scaled = s.fit_transform(df_temp)

    # Create PCA reduced dataset
    pca = PCA(20)
    pca_train = pca.fit_transform(X_scaled)
    
    # Transform data with UMAP
    transf = umap.UMAP(n_neighbors=10)
    X_umap = transf.fit_transform(pca_train)

    # Plot Umap's with target colorization
    print('Plotting', key)
    fig, ax = plt.subplots(nrows=1, ncols=7, figsize=(21, 3))
    for i, c in enumerate(targets.columns):
        ax[i].scatter(X_umap[:, 0], X_umap[:, 1], s=1, c=targets[c].values, cmap='Spectral')
        ax[i].set_title(c)
        ax[i].axis('off')
    plt.show()