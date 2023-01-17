import PIL

import os, re, time, tqdm

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm



from scipy import stats

import shap



from sklearn import utils, model_selection

import lightgbm as lgb



import matplotlib.pyplot as plt

import cv2

import pydicom

import tensorflow as tf
img_dim = 128

min_layers = 200

quantiles_thrs = [0.1, 0.3, 0.5, 0.7, 0.9]
def read_tomography(folder, dim=128, norm=False):

    files = np.array(os.listdir(folder))

    files = files[np.argsort([int(f.split('.')[0]) for f in files])]

    images = []

    for file in files:

        dicom = pydicom.dcmread(os.path.join(folder, file))

        try:

            images += [cv2.resize(dicom.pixel_array, (dim, dim)).reshape(dim, dim)]

        except RuntimeError:

            pass

    images = np.array(images)

    if np.ndim(images) <= 2:

        images = np.nan * np.ones((1, dim, dim))

    if norm:

        images = (images - images.min()) / (images.max() - images.min())

    return np.array(images)
df_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

df_test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
def movement(x, axis=None):

    return np.mean(np.abs(np.diff(x, axis=axis)), axis=axis)





def compl_axis(ax):

    assert isinstance(ax, list) or isinstance(ax, int)

    axs = list()

    if isinstance(ax, list):

        for a in ax:

            if isinstance(a, int):

                axs.append(a)

            elif isinstance(a, tuple) or isinstance(a, list):

                axs.extend(list(a))

    else:

        axs = [ax]

    return tuple([i for i in range(3) if i not in axs])





def build_fn2(fns, ax):

    if isinstance(ax, int) or isinstance(ax, tuple):

        ax = [ax]

    assert len(fns) == 2

    assert len(ax) == 1

    def fn(x):

        x = np.expand_dims(fns[0](x, axis=ax[0]), ax[0])

        c_ax = compl_axis(ax)

        # x = fns[1](x, axis=c_ax)

        x = fns[1](np.ravel(x), axis=0)

        x = float(np.squeeze(x))

        return x

    return fn





def build_fn3(fns, axs):

    assert len(fns) == 3

    assert len(axs) == 2

    def fn(x):

        x = np.expand_dims(fns[0](x, axis=axs[0]), axs[0])

        x = np.expand_dims(fns[1](x, axis=axs[1]), axs[1])

        x = fns[2](np.ravel(x), axis=0)

        x = float(np.squeeze(x))

        return x

    return fn
features_funcs = {

    'mean': np.mean,

    'std': np.std,

}

base_aggrs = [('mean', np.mean), ('std', np.std)]

aggrs = base_aggrs + [('kurtosis', stats.kurtosis), ('skew', stats.skew)]

aggrs_mono = aggrs + [('move', movement)]

axis = [('top_down', 0), ('left_right', 1), ('front_rear', 2)]

planes = [('frontal', (0, 1)), ('sagittal', (0, 2)), ('trasversal', (1, 2))]

# plane-axis

for pl_name, pl in planes:

    for fn0_name, fn0 in base_aggrs:

        for fn1_name, fn1 in aggrs_mono:

            if not (fn0_name == 'mean' and fn1_name == 'mean'):

                features_funcs['%s__%s_%s' % (fn1_name, pl_name, fn0_name)] = build_fn2([fn0, fn1], [pl])

# axis-plane

for ax_name, ax in axis:

    for fn0_name, fn0 in base_aggrs:

        for fn1_name, fn1 in aggrs:

            if not (fn0_name == 'mean' and fn1_name == 'mean'):

                features_funcs['%s__%s_%s' % (fn1_name, pl_name, fn0_name)] = build_fn2([fn0, fn1], [ax])

# axis-axis-axis

for ax0_name, ax0 in axis:

    for ax1_name, ax1 in axis:

        for fn0_name, fn0 in base_aggrs:

            for fn1_name, fn1 in aggrs_mono:

                for fn2_name, fn2 in aggrs_mono:

                    if ax0 != ax1 and (not (fn0_name == 'mean' and fn1_name == 'mean')) and (not (fn1_name == 'mean' and fn2_name == 'mean')):

                        features_funcs['%s__%s_%s__%s_%s' % (fn2_name, ax1_name, fn1_name, ax0_name, fn0_name)] = build_fn3([fn0, fn1, fn2], [ax0, ax1])

print("%d features generated" % len(features_funcs))
features = pd.DataFrame()

for i, patient in enumerate(tqdm(df_train.Patient.drop_duplicates(), desc='Patient loop')):

    folder = os.path.join('../input/osic-pulmonary-fibrosis-progression/train', patient)

    images = read_tomography(folder, dim=img_dim, norm=True)

    depth = images.shape[0]

    while images.shape[0] > min_layers:

        images = images[::2]

    f = {'Patient': patient, 'imgs_num': len(os.listdir(folder))}

    for key in features_funcs.keys():

        f['img__' + key] = features_funcs[key](images)

        for thr in quantiles_thrs:

            f['img__' + key + "__thr%2d" % int(100 * thr)] = features_funcs[key](images >= np.quantile(images, thr))

    features = features.append(pd.DataFrame(f, index=[i]))

    pass
plt.figure(figsize=(6, 12))

plt.subplot(2, 1, 1)

plt.imshow(images[0])

plt.subplot(2, 1, 2)

m = images[0].mean(0); plt.plot((m - m.mean()) / m.std(), color='tab:blue')

m = images[0].std(0); plt.plot((m - m.mean()) / m.std(), color='tab:orange')

m = movement(images[0], 0); plt.plot((m - m.mean()) / m.std(), color='tab:green')
features.to_csv("all_features.csv", index=False)

features.head()
features_test = pd.DataFrame()

for i, patient in enumerate(tqdm(df_test.Patient.drop_duplicates(), desc='Patient loop')):

    folder = os.path.join('../input/osic-pulmonary-fibrosis-progression/test', patient)

    images = read_tomography(folder, dim=img_dim, norm=True)

    depth = images.shape[0]

    while images.shape[0] > min_layers:

        images = images[::2]

    f = {'Patient': patient, 'imgs_num': len(os.listdir(folder))}

    for key in features_funcs.keys():

        f['img__' + key] = features_funcs[key](images)

        for thr in quantiles_thrs:

            f['img__' + key + "__thr%2d" % int(100 * thr)] = features_funcs[key](images >= np.quantile(images, thr))

    features_test = features_test.append(pd.DataFrame(f, index=[i]))

    pass
features_test.to_csv("all_features_test.csv", index=False)

features_test.head()
features = pd.read_csv("all_features.csv")

print("Features has %d columns" % features.shape[1])
MAX_CORRELATION = 0.995



corr = features.iloc[:100].replace(np.nan, features.iloc[:1000].mean(0)).corr()

to_check = []

for f0 in corr.index:

    f1s = corr.loc[f0].index[corr.loc[f0] > MAX_CORRELATION]

    for f1 in f1s:

        if f0 != f1:

            to_check += [f0, f1]

to_check = list(set(to_check))

corr = features[to_check].replace(np.nan, features[to_check].mean(0)).corr()

duplicated = []

for i0, f0 in enumerate(tqdm(corr.index)):

    f1s = corr.loc[f0].index[(corr.loc[f0] > MAX_CORRELATION) & (np.array(range(corr.shape[1])) > i)] 

    for f1 in f1s:

        if f0 != f1:

            duplicated += [(f0, f1)]

print("%d (almost) duplicated features found" % len(duplicated))

features = features.drop([d for _, d in duplicated], axis=1)

print("Features has %d columns" % features.shape[1])

features.to_csv("features_no_duplicates.csv", index=False)
features = pd.read_csv("features_no_duplicates.csv")

train_base = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv').sort_values(['Patient', 'Weeks'])

train = train_base.copy()

ref = train.copy()[['Weeks', 'Patient', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']]

tar = train.copy()[['Weeks', 'Patient', 'FVC']]

train = pd.merge(ref, tar, on='Patient', suffixes=['_base', ''])

train['Week_passed'] = train['Weeks'] - train['Weeks_base']

train = train[train['Week_passed'] != 0]

base_FE = ['FVC_base', 'Week_passed', 'Age', 'Percent']

train.head()
metric = 'l1'

selected_features = list(base_FE)

res_features = list(features.drop('Patient', 1).columns)

params = {'n_estimators': 250, 'learning_rate': 0.2, 'reg_alpha': 1.0, 'reg_lambda': 1.0, 'subsample': 0.9}

X, y = train.merge(features, on='Patient', how='left'), train.FVC

idxT, idxV = list(model_selection.GroupKFold(10).split(X, y, train.Patient))[0]

X_train, y_train = X.iloc[idxT], y.iloc[idxT]

X_valid, y_valid = X.iloc[idxV], y.iloc[idxV]

interesting_ones = []

to_remove = []

for _ in range(100):

    model = lgb.LGBMRegressor(**params)

    model.fit(X_train[selected_features], y_train, eval_set=[(X_valid[selected_features], y_valid)], eval_metric=metric, verbose=0)

    baseline = min(model.evals_result_['valid_0'][metric])

    print("Baseline\t\t%.4f" % baseline)

    best_val = baseline

    best_f = None

    trange = tqdm(res_features)

    for f in trange:

        if f in selected_features or f in to_remove:

            continue

        model.fit(X_train[selected_features + [f]], y_train, eval_set=[(X_valid[selected_features + [f]], y_valid)], eval_metric=metric, verbose=0)

        val = min(model.evals_result_['valid_0'][metric])

        if val < baseline:

            interesting_ones.append(f)

        if val < best_val:

            best_val = val

            best_f = f

            trange.set_description("Found %.4f" % best_val)

            

    if best_f is not None:

        print("+%s\t\t%.4f" % (best_f, best_val))

        selected_features.append(best_f)

    else:

        print("STOP")

        break
features[['Patient'] + [f for f in selected_features if f in features.columns]].to_csv("features_selected.csv", index=False)

features[['Patient'] + [f for f in list(set(interesting_ones)) if f in features.columns]].to_csv("features_interesting.csv", index=False)
params = {'learning_rate': 0.02, 'reg_lambda': 50, 'reg_alpha': 50, 'n_estimators': 1000, 'subsample': 0.5}

model = lgb.LGBMRegressor(**params)

model.fit(X_train[base_FE], y_train, eval_set=[(X_valid[base_FE], y_valid)], eval_metric=metric, verbose=0)

model_imgs = lgb.LGBMRegressor(**params)

model_imgs.fit(X_train[selected_features], y_train, eval_set=[(X_valid[selected_features], y_valid)], eval_metric=metric, verbose=0)

plt.plot(model.evals_result_['valid_0'][metric], color='tab:blue', label='base data')

plt.plot(model_imgs.evals_result_['valid_0'][metric], color='tab:red', linestyle='--', label='base data + imgs')

plt.yscale('log'); plt.legend();

print("%.6f ==> %.6f" % (min(model.evals_result_['valid_0'][metric]), min(model_imgs.evals_result_['valid_0'][metric])))