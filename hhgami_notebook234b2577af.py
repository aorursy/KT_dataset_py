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
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
train_target_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
train_target_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train['dataset'] = "train"
test['dataset'] = "test"
train_target_all = pd.concat([train_target_scored, train_target_nonscored], axis = 1)
train_target_all.head()
train_target_all.shape
import seaborn as sns
import matplotlib.pyplot as plt
train['activation'] = train_target_all.sum(axis = 1)
sns.countplot(train['activation'], hue = train.cp_type)
plt.legend()
plt.show()
#Split columns
columns = train.columns.to_list()
g_list = [i for i in columns if i.startswith('g-')]
c_list = [i for i in columns if i.startswith('c-')]
#PCA for gene expression columns
from sklearn.decomposition import PCA
pca = PCA(n_components = len(g_list))
pca.fit(train[g_list])
#Contribution of principal components
variance_ratio = pd.DataFrame()
variance_ratio['ratio'] = pca.explained_variance_ratio_
variance_ratio['cumsum'] = np.cumsum(pca.explained_variance_ratio_)
variance_ratio.index = ["P{}".format(x+1) for x in range(len(g_list))] 

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

plt.figure(figsize = (20,6))
sns.barplot(data = variance_ratio[:80], x = variance_ratio.index[:80], y = 'ratio')
sns.pointplot(data = variance_ratio[:80], x = variance_ratio.index[:80], y = 'cumsum')
plt.title("Contribution of first 80 components")
plt.ylabel("contribution")
plt.xlabel("components")
plt.grid()
plt.show()
#PCA scores
feature_g = pca.transform(train[g_list])
PCA_g = pd.DataFrame(feature_g, columns=["gPC{}".format(x+1) for x in range(len(g_list))], index = train.index)
PCA_g.head()
#Scatter plots in principal components space
plt.figure(figsize=(6,6))
sns.scatterplot(x = feature_g[:,0], y = feature_g[:,1], hue = train['activation'], palette = "deep")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
#PCA for cell viability columns
pca = PCA(n_components = len(c_list))
pca.fit(train[c_list])
#Contribution of principal components
variance_ratio = pd.DataFrame()
variance_ratio['ratio'] = pca.explained_variance_ratio_
variance_ratio['cumsum'] = np.cumsum(pca.explained_variance_ratio_)
variance_ratio.index = ["PC{}".format(x+1) for x in range(len(c_list))]

plt.figure(figsize = (30,6))
sns.barplot(data = variance_ratio[:50], x = variance_ratio.index[:50], y = 'ratio')
sns.pointplot(data = variance_ratio[:50], x = variance_ratio.index[:50], y = 'cumsum')
plt.title("Contribution of first 50 components")
plt.xlabel("components")
plt.ylabel("contribution")
plt.grid()
plt.show()
#PCA scores
feature_c = pca.transform(train[c_list])
PCA_c = pd.DataFrame(feature_c, columns=["cPC{}".format(x+1) for x in range(len(c_list))], index = train.index)
PCA_c.head()
#Scatter plots in principal components space
plt.figure(figsize=(6,6))
sns.scatterplot(x = feature_c[:,0], y = feature_c[:,1], hue = train['activation'], palette = "deep")
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
activation_2 = train_all['activation'] == 2
train_target_2 = train_target[activation_2].iloc[:, 1:]
count = pd.DataFrame(train_target_2.sum(axis = 0), index = train_target_2.columns)
rows_to_drop = count.index[count[0] == 0]
count = count.drop(rows_to_drop)
plt.figure(figsize=(6,18))
sns.barplot(x = count[0], y = count.index)
plt.show()
train_target_2.shape
train_target_2
count.index.shape
sns.heatmap(train_target_2[count.index].corr())
plt.show()
print("missing values in train dataset:", train.isnull().sum().sum())
#Categorical columns
object_cols = ["cp_type", "cp_dose"]
#Encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in object_cols:
    df[col] = label_encoder.fit_transform(df[col])
#cp_time scale is changed to from 0 to 1.
s = df['cp_time']
df['cp_time'] = (s/s.min())*(1/3)
df['cp_time'].describe()
#First 10 PCs for cell viavility and first 80 PCs for gene expression are selected as features.
X = pd.concat([df.iloc[:len(train), :3], PCA_g.iloc[:len(train),:80], PCA_c.iloc[:len(train),:10]], axis = 1)
y = train_target
X_test = pd.concat([df.iloc[len(train):, :3], PCA_g.iloc[len(train):,:80], PCA_c.iloc[len(train):,:10]], axis = 1)

# drop id col
X = X.iloc[:,0:].to_numpy()
X_test = X_test.iloc[:,0:].to_numpy()
y = y.iloc[:,0:].to_numpy() 
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators = 50, max_depth = 10, random_state = 0)
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

test_preds = np.zeros((test.shape[0], y.shape[1]))
oof_preds = np.zeros(y.shape)
oof_losses = []

cv = KFold(n_splits = 5)
for fold_id, (train_index, valid_index) in enumerate(cv.split(X, y)):
    print('Starting fold:', fold_id)
    X_train = X[train_index,:]
    X_val = X[valid_index,:]
    y_train = y[train_index,:]
    y_val = y[valid_index,:]
    
    ctl_mask = X_train[:,0]==0
    X_train = X_train[~ctl_mask,:]
    y_train = y_train[~ctl_mask]
    
    clf.fit(X_train, y_train)
    val_preds = clf.predict(X_val)
    oof_preds[valid_index] = val_preds
    
    loss = log_loss(np.ravel(y_val), np.ravel(val_preds))
    oof_losses.append(loss)
    preds = clf.predict(X_test)
    test_preds += preds / 5

print(oof_losses)
print('Mean OOF loss', np.mean(oof_losses))
print('STD OOF loss', np.std(oof_losses))
# set control train preds to 0
control_mask = train['cp_type']=='ctl_vehicle'
oof_preds[control_mask] = 0

print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))
# set control test preds to 0
control_mask = test['cp_type']=='ctl_vehicle'

test_preds[control_mask] = 0
test_preds
test_preds.shape
sub.shape
# create the submission file
sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
sub.iloc[:,1:] = np.array(test_preds)
sub.to_csv('submission.csv', index=False)