import h5py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
import scipy as sp
import random
import nilearn as nl
from nilearn import datasets
from nilearn import plotting
from nilearn import image
import nibabel as nib
import nilearn.plotting as nlplt
import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn import preprocessing
import category_encoders as ce
from sklearn.metrics import mean_squared_error

import torch

import lightgbm as lgb
from glob import glob

import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('/kaggle/input/trends-assessment-prediction/train_scores.csv').sort_values(by='Id')

loadings = pd.read_csv('/kaggle/input/trends-assessment-prediction/loading.csv')

fnc = pd.read_csv('/kaggle/input/trends-assessment-prediction/fnc.csv')

sample = pd.read_csv('/kaggle/input/trends-assessment-prediction/sample_submission.csv')

reveal = pd.read_csv('/kaggle/input/trends-assessment-prediction/reveal_ID_site2.csv')

ICN = pd.read_csv('/kaggle/input/trends-assessment-prediction/ICN_numbers.csv')
!wget https://github.com/Chaogan-Yan/DPABI/raw/master/Templates/ch2better.nii
mat = h5py.File('/kaggle/input/trends-assessment-prediction/fMRI_train/10046.mat','r')
mat.keys()
sample = mat['SM_feature']
array = sample[()]
array.shape
print(array.min(),array.max(),array.mean())
mat, ax = plt.subplots(1,4)
mat.set_size_inches(25, 10)
for i in range(4):
    Temp = array[i*10, :, 10, :] !=0  
    ax[i].imshow(Temp)
plt.show()
motor_images = datasets.fetch_neurovault_motor_task()
img = motor_images.images[0]
nii_loc = "/kaggle/input/trends-assessment-prediction/fMRI_mask.nii"
nii_loc2 = "/kaggle/input/trends-assessment-prediction/fMRI_train/10009.mat"
niiplot = plotting.plot_glass_brain(img)
niiplot
maskni = nl.image.load_img(nii_loc)
subjectimage = nl.image.new_img_like(nii_loc, array, affine=maskni.affine, copy_header=True)
smri = 'ch2better.nii'
num_components = subjectimage.shape[-1]
grid_size = int(np.ceil(np.sqrt(num_components)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*10, grid_size*10))
[axi.set_axis_off() for axi in axes.ravel()]
row = -1
for i, cur_img in enumerate(nl.image.iter_img(subjectimage)):
    col = i % grid_size
    if col == 0:
        row += 1
    nlplt.plot_stat_map(cur_img, bg_img=smri, title="IC %d" % i, axes=axes[row, col], threshold=3, colorbar=False)
train.isnull().sum()
reveal.head()
ICN.head()
fnc.head()
train_ids = sorted(loadings[loadings['Id'].isin(train.Id)]['Id'].values)
test_ids = sorted(loadings[~loadings['Id'].isin(train.Id)]['Id'].values)
predictions = pd.DataFrame(test_ids, columns=['Id'], dtype=str)
features = ('age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2')
data = pd.merge(loadings, train, on='Id').dropna()
X_train = data.drop(list(features), axis=1).drop('Id', axis=1)
y_train = data[list(features)]
X_test = loadings[loadings.Id.isin(test_ids)].drop('Id', axis=1)
X_test.head()
y_train.head()
def plot_bar(df, feature, title='', show_percent = False, size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8, palette='Set2')

    plt.title(title)
    if show_percent:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center", rotation=45) 
    plt.xlabel(feature, fontsize=12, )
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()
plot_bar(y_train, 'age', 'age count and %age plot', show_percent=True, size=4)
def plot_bar(df, feature, title='', show_percent = False, size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8, palette='Set2')

    plt.title(title)
    if show_percent:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center", rotation=45) 
    plt.xlabel(feature, fontsize=12, )
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()
### Age count Distribution
for col in y_train.columns[2:]:
    plot_bar(y_train, col, f'{col} count plot', size=4)
plt.figure(figsize = (12, 8))
sns.heatmap(y_train.corr(), annot = True, cmap="RdYlGn")
plt.yticks(rotation=0) 

plt.show()
temp_data =  loadings.drop(['Id'], axis=1)

plt.figure(figsize = (20, 20))
sns.heatmap(temp_data.corr(), annot = True, cmap="RdYlGn")
plt.yticks(rotation=0) 

plt.show()
model = RandomForestRegressor(
    max_depth=15,
    min_samples_split=8,
    min_samples_leaf=7
)
cv = KFold(n_splits = 7, shuffle=True, random_state=35)
grid = {
    'n_estimators':[5,10,20,100]
}
gs = GridSearchCV(model, grid, n_jobs=-1, cv=cv, verbose=1, scoring='neg_mean_absolute_error')
best_models = {}
for col in features:
    gs.fit(X_train, y_train[col])
    best_models[col] = gs.best_estimator_
    print(gs.best_score_)
for col in features:
    predictions[col] = best_models[col].predict(X_test)
def make_sub(predictions):
    features = ('age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2')
    _columns = (0,1,2,3,4)
    tests = predictions.rename(columns=dict(zip(features, _columns)))
    tests = tests.melt(id_vars='Id',value_vars=_columns,value_name='Predicted')
    tests['target'] = tests.variable.map(dict(zip(_columns, features)))
    tests['Id_'] = tests[['Id', 'target']].apply(lambda x: '_'.join((str(x[0]), str(x[1]))), axis=1)
  
    return tests.sort_values(by=['Id', 'variable'])\
              .drop(['Id', 'variable', 'target'],axis=1)\
              .rename(columns={'Id_':'Id'})\
              .reset_index(drop=True)\
              [['Id', 'Predicted']]
sub = make_sub(predictions)
sub.to_csv('submission.csv', index=False)