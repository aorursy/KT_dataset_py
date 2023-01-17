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
# importing packages, libraries
from tqdm import tqdm
import numpy as np 
import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.sparse import hstack
from matplotlib import pyplot as plt
import seaborn as sns
# Reading dataset
train = pd.read_csv('/kaggle/input/minor-project-2020/train.csv').fillna(' ')
test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv').fillna(' ')
train.info(), test.info()
train["target"].value_counts()
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(train.drop(columns=["id", "target"]).values, train["target"])
labels=pd.DataFrame(y_res)
X=pd.DataFrame(X_res)
train = pd.merge(left=X, left_index=True, right=labels, right_index=True, how='inner')
train.rename(columns={"0_x": 0, "0_y": "target"}, inplace=True)
train
train["target"].value_counts()
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in train.items():
    if (index==14):
        break
    else:
        sns.distplot(v, ax=axs[index])
        index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
#BoxPlots
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in train.items():
    if (index==14):
        break
    else:
        sns.boxplot(x=k, data=train, ax=axs[index])
        index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# calculating skewness of each numerical column
num_col=train.dtypes[train.dtypes!="object"].index             # numerical columns
skew_col=train[num_col].skew().sort_values(ascending=False)    # calculating skew value
skew_col
# combining X_train and X_test before applying scaling, transformations to ensure uniformity in data fed into model 
X_train=train.drop(["target"], axis=1).values
X_test=test.drop(["id"], axis=1).values
y_train=y_res
X=np.vstack((X_train, X_test))
X_train.shape, X_test.shape, X.shape
# scaling to fit boxcox criteria
from sklearn.preprocessing import MinMaxScaler

scaler2 = MinMaxScaler((0.001, 1)).fit(X)
X_scaled = scaler2.transform(X)

dataset=pd.DataFrame(X_scaled)
#BoxPlots
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in dataset.items():
    if (index==14):
        break
    else:
        sns.boxplot(x=k, data=dataset, ax=axs[index])
        index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
dataset
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in dataset.items():
    if (index==14):
        break
    else:
        sns.distplot(v, ax=axs[index])
        index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# calculating skewness of each numerical column of entire dataset
num_col=dataset.dtypes[dataset.dtypes!="object"].index             # numerical columns
skew_col=dataset[num_col].skew().sort_values(ascending=False)    # calculating skew value
skew_col
# boxcox
modset=dataset.copy()
from scipy import stats
for col in tqdm(modset.columns):
    modset[col]=stats.boxcox(modset[col])[0]
# calculating skewness after transformation
num_col=modset.dtypes[modset.dtypes!="object"].index             # numerical columns
skew_col=modset[num_col].skew().sort_values(ascending=False)    # calculating skew value
skew_col

# skewness reduced quite a bit
# dropping 45, 44, 43  because highly skewed even after boxcox
#modset.drop(columns=[43, 44, 45], inplace=True)
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in modset.items():
    if (index==14):
        break
    else:
        sns.distplot(v, ax=axs[index])
        index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
for col in modset.columns:
    if (modset[col].isna().values.any()):
        print("F")
# no nans :)
# saving transformed unscaled data
X_mod=modset.values
with open("./X-oversampled-corpus-unskew-boxcox-1-unscaled.pickle", "wb") as f:
    pickle.dump(X_mod, f)
#saving labels
target=train["target"].values
with open("./y-oversampled-corpus-unskew-boxcox-1-scaled.pickle", "wb") as f:
    pickle.dump(target, f)
#loading transformed unscaled data
with open("./X-oversampled-corpus-unskew-boxcox-1-unscaled.pickle", "rb") as f:
    X_mod=pickle.load(f)
modset=pd.DataFrame(X_mod)
# loading labels
with open("./y-oversampled-corpus-unskew-boxcox-1-scaled.pickle", "rb") as f:
    target=pickle.load(f)
# standard scaling
X_mod=modset.values
from sklearn.preprocessing import MinMaxScaler

scaler3 = MinMaxScaler((0.001, 1)).fit(X_mod)
X_scaled3 = scaler3.transform(X_mod)

modset=pd.DataFrame(X_scaled3)
modset
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=85)
X_sel=selector.fit_transform(X_scaled3[:1596998], target)
X_sel.shape
dropped_features=[]
bool_features=selector.get_support()
for i in range(0, len(bool_features)):
    if (bool_features[i]==False):
        dropped_features+=[i]
dropped_features

modset.drop(columns=[item for item in dropped_features], inplace=True)
target=train["target"].values
train_label = pd.merge(left=modset[:1596998], left_index=True, right=pd.DataFrame(target).rename(columns={0:"target"}),
                       right_index=True, how='inner')
train_label
# high corr among features
high_corr=[]
for key in corr.keys():
    for col in corr.columns:
        if (key!=col and np.abs(corr[col][key])>=0.7 and ((col, key) not in high_corr)):
            high_corr+=[(key, col)]
high_corr
low_corr=[]
for key in corr.keys():
        if (key!="target" and np.abs(corr["target"][key])<0.001):
            low_corr+=[(key, "target")]
low_corr
# removing high_corr col, not removing low_corr columns
train_label.drop(columns=[item[0] for item in high_corr], inplace=True)
final_X_train=train_label.drop(columns=["target"]).values
final_y_train=train_label["target"].values
# Linear classifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
# following are best params obtained from grid search
model=SGDClassifier(alpha=1e-5, eta0=1, l1_ratio=0, learning_rate='adaptive',
              loss='log', n_jobs=-1, penalty='elasticnet', verbose=1,  n_iter_no_change=10)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
%%time
cv_results = cross_val_score(model, final_X_train, final_y_train, cv=skf, scoring='roc_auc')
print(cv_results, cv_results.mean())
# not using mp-1 files
#!cp -R /kaggle/input/mp-1-files/grid-sgd-v1.csv ./
#!cp -R /kaggle/input/mp-1-files/grid-sgd-v1.pickle ./
model.fit(final_X_train, final_y_train)
with open("./grid-sgd-v1.pickle", "wb") as f:
    pickle.dump(model, f)
"""
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

alpha=[10**i for i in [-4, -5, -6]]
l1_ratio=[0, 0.3, 0.7, 1]
loss=["log", "modified_huber"]

# Parameter grid: dictionary of parameter list ranges
param_grid={
        "alpha" : alpha,
        "penalty": ["elasticnet"],
        "l1_ratio": l1_ratio,
        "verbose": [1],
        "learning_rate": ["adaptive"],
        "loss": loss,
        "eta0": [1],
        "n_jobs": [-1],
        "shuffle": [True],
}

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid=GridSearchCV(SGDClassifier(), param_grid, cv=skf, n_jobs=-1, scoring="roc_auc", verbose=10, return_train_score=True)
"""""""
"""
%%time
grid_result=grid.fit(final_X_train, final_y_train)
"""
"""
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
"""
"""
# save the gridsearch model
import pickle
with open(root_path+"grid-sgd-v1.pickle", "wb") as f:
    pickle.dump(grid, f)

model=grid.best_estimator_
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
""""""
test_df=modset[1596998:]
#test_df=test_df.drop(columns=[item[1] for item in high_corr])
test_df
preds=model.predict_proba(test_df.values)
preds
posPreds=[]
for y in preds:
    posPreds+=[y[1]]
posPreds
# Generating predictions csv
sub = pd.read_csv('/kaggle/input/minor-project-2020/test.csv').fillna(' ')
my_submission = pd.DataFrame({'id': sub.id, 'target': posPreds})
my_submission.to_csv('grid-sgd-v6-sel70.csv', index=False)