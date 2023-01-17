# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import random
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.metrics import *
from scipy.stats import pearsonr
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/lish-moa/train_features.csv')
df_test = pd.read_csv('../input/lish-moa/test_features.csv')
df_train_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
df_train_unscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

print(df_train.head())
print(df_train_scored.head())
# check For missing values 
df_train.isnull().sum().sum()
# check For missing values 
df_test.isnull().sum().sum()
# check for target sparsity
scored = df_train_scored.drop(columns = ["sig_id"] , axis = 1)
# non zero target varaibles
print((scored.to_numpy()).sum()/(scored.shape[0]*scored.shape[1])*100 , "%")
print("Shape of {}: {}".format("training data",df_train.shape))
print("Shape of {}: {}".format("target data",df_train_scored.shape))
print("Shape of {}: {}".format("testing data",df_test.shape))
df_train.describe()
df_train_unscored.describe()
df_train_scored.describe()
plt.rcParams['figure.figsize']=(20,5)
fig, ax = plt.subplots(1,3)
sns.countplot(df_train["cp_type"], ax = ax[0])
sns.countplot(df_train["cp_time"], ax= ax[1])
sns.countplot(df_train["cp_dose"], ax = ax[2])

fig, (cp_type_bar, cp_dose_bar) = plt.subplots(nrows=1, ncols=2, figsize=[12, 6])

# plot frequency of cp_type
cp_type_training_count = df_train['cp_type'].value_counts()
cp_type_test_count = df_test['cp_type'].value_counts()
cp_type_label = cp_type_training_count.index
cp_type_width = 1.0
cp_type_bar.bar([0, 3], cp_type_training_count, width=cp_type_width)
cp_type_bar.bar([1, 4], cp_type_test_count, width=cp_type_width)
cp_type_bar.set_xticks([0.5, 3.5])
cp_type_bar.set_xticklabels(cp_type_label)
cp_type_bar.set_title('Frequency of cp_type in training')

# plot frequency of cp_dose
cp_dose_training_count = df_train['cp_dose'].value_counts()
cp_dose_test_count = df_test['cp_dose'].value_counts()
cp_dose_label = cp_dose_training_count.index
cp_dose_width = 1.0
cp_dose_bar.bar([0, 3], cp_dose_training_count, width = cp_dose_width)
cp_dose_bar.bar([1, 4], cp_dose_test_count, width = cp_dose_width)
cp_dose_bar.set_xticks([0.5, 3.5])
cp_dose_bar.set_xticklabels(cp_dose_label)
cp_dose_bar.set_title('Frequency of cp_dose in training')

plt.show()
df_train_full = pd.merge(df_train_scored,df_train_unscored,on = "sig_id", how = 'inner')
df_train_full
# list the columns 
# list(features)
# get all the gene features and cell features
common  = ['sig_id',
 'cp_type',
 'cp_time',
 'cp_dose']


genes = list(filter(lambda x : "g-" in x  , list(df_train)))

cells = list(filter(lambda x : "c-" in x  , list(df_train)))
#Let's take a look at the distribution for some genes
nrows, ncols = 3, 3
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=[36,24])
fig.tight_layout(pad=12.0)
cmap = plt.cm.get_cmap("tab10")
colors = cmap.colors
# plot pdf for 9 random g- features
for i in range(nrows):
    for j in range(ncols):
        feature = random.randint(0, 771)
        axis = ax[i][j]
        axis.hist(df_train[f'g-{feature}'], bins=100, density=True, color=colors[2 * i + j])
        axis.set_title(f'pdf for g-{feature}', {'fontsize': 32})
        axis.set_xlabel("Numerical value in training set", {'fontsize': 18})
        axis.set_ylabel("Probability density", {'fontsize': 18})

plt.show()
# Some slightly skewed data for features in g
nrows, ncols = 2, 2
fig, ax = plt.subplots(figsize=[24, 18], nrows=nrows, ncols=ncols)
skewed_g = [['g-744', 'g-123'], ['g-489', 'g-644']]
cmap = plt.cm.get_cmap("Set2")
colors = cmap.colors
for i in range(nrows):
    for j in range(ncols):
        axis = ax[i][j]
        axis.hist(df_train[skewed_g[i][j]], bins=100, density=True, color=colors[2 * i + j])
        axis.set_title(f'pdf for {skewed_g[i][j]}')
        axis.set_xlabel("Numerical value in training set", {'fontsize': 18})
        axis.set_ylabel("Probability density", {'fontsize': 18})
plt.show()
#g-744,g-123 g-489, g-644g-23, #g-644, g-413, g-307, g-238
# More slightly skewed data for features in g
nrows, ncols = 2, 2
fig, ax = plt.subplots(figsize=[24, 18], nrows=nrows, ncols=ncols)
skewed_g = [['g-23', 'g-413'], ['g-307', 'g-238']]
cmap = plt.cm.get_cmap("Set2")
colors = cmap.colors
for i in range(nrows):
    for j in range(ncols):
        axis = ax[i][j]
        axis.hist(df_train[skewed_g[i][j]], bins=100, density=True, color=colors[2 * i + j])
        axis.set_title(f'pdf for {skewed_g[i][j]}', {'fontsize': 24})
        axis.set_xlabel("Numerical value in training set", {'fontsize': 16})
        axis.set_ylabel("Probability density", {'fontsize': 16})
plt.show()
# some stats plot for genes. Find distribution of max/min/mean/std across all gene columns.
fig, axs = plt.subplots(ncols=2 , nrows = 2 , figsize=(13,13))
sns.distplot(df_train[genes].max(axis =1) ,color="b",hist=False, kde_kws={"shade": True}, ax=axs[0][0] ).set(title = 'max')
sns.distplot(df_train[genes].min(axis =1) ,color="r",hist=False, kde_kws={"shade": True}, ax=axs[0][1] ).set(title = 'min')
sns.distplot(df_train[genes].mean(axis =1), color="g",hist=False, kde_kws={"shade": True}, ax=axs[1][0] ).set(title = 'mean')
sns.distplot(df_train[genes].std(axis =1) ,color="y",hist=False, kde_kws={"shade": True}, ax=axs[1][1] ).set(title = 'sd')
plt.show()
#show histogram plots for randomly chosen cells
nrows, ncols = 3, 3
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=[36,24])
fig.tight_layout(pad=12.0)
cmap = plt.cm.get_cmap("tab10")
colors=cmap.colors
# plot pdf for 9 random c- features
for i in range(nrows):
    for j in range(ncols):
        feature = random.randint(0, 99)
        axis = ax[i][j]
        axis.hist(df_train[f'c-{feature}'], bins=100, density=True, color=colors[i*2+j])
        axis.set_title(f'pdf for c-{feature}', {'fontsize': 32})
        axis.set_xlabel("Numerical value in training set", {'fontsize': 18})
        axis.set_ylabel("Probability density", {'fontsize': 18})

plt.show()
# some stats plot for cell viability. Find distribution of max/min/mean/std across all cell columns. 
fig, axs = plt.subplots(ncols=2 , nrows = 2 , figsize=(13,13))
sns.distplot(df_train[cells].max(axis =1) ,color="b",hist=False, kde_kws={"shade": True}, ax=axs[0][0] ).set(title = 'max')
sns.distplot(df_train[cells].min(axis =1) ,color="r",hist=False, kde_kws={"shade": True}, ax=axs[0][1] ).set(title = 'min')
sns.distplot(df_train[cells].mean(axis =1), color="g",hist=False, kde_kws={"shade": True}, ax=axs[1][0] ).set(title = 'mean')
sns.distplot(df_train[cells].std(axis =1) ,color="y",hist=False, kde_kws={"shade": True}, ax=axs[1][1] ).set(title = 'sd')
plt.show()
target  = df_train_scored.drop(['sig_id'] , axis =1)

fig, ax = plt.subplots(figsize=(9,9))
ax = sns.countplot(target.sum(axis =1), palette="Set2")
total = float(len(target))

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.4f}%'.format((height/total)*100),
            ha="center") 

plt.show()
## counts per target class- 
sns.kdeplot(target.sum() , shade = True , color = "b")
top_targets = pd.Series(target.sum()).sort_values(ascending=False)[:5]
bottom_targets = pd.Series(target.sum()).sort_values()[:5]
fig, axs = plt.subplots(figsize=(9,9) , nrows=2)
sns.barplot(top_targets.values , top_targets.index , ax = axs[0] ).set(title = "Top five targets")
sns.barplot(bottom_targets.values , bottom_targets.index, ax = axs[1] ).set(title = "bottom five targets")
plt.show()
cols = pd.DataFrame({'value': [1 for i in list(target) ]} , index = [i.split('_')[-1] for i in list(target)] )
cols_top_5 = cols.groupby(level=0).sum().sort_values(by = 'value' , ascending = False)[:5]

fig, ax = plt.subplots(figsize=(9,9))

sns.barplot(x = cols_top_5.value , y = cols_top_5.index , palette="Set2" , orient='h')


for p in ax.patches:
    width = p.get_width()
    plt.text(8+p.get_width(), p.get_y()+0.55*p.get_height(),
             '{:1.4f}%'.format((width /206 )*100), # total 206 columns
             ha='center', va='center')

plt.show()
print("Top five suffixes constitue for about ", list(cols_top_5.sum()/cols.sum().values)[0]*100 , "%")
# RobustScalar transforms the feature vector by subtracting the median and then dividing by the interquartile range (25% - 75%)
#note: id column is dropped here
df_copy = df_train.copy(deep=True)
df_id = df_train["sig_id"]
df_copy.drop("sig_id", axis = 1, inplace = True)
df_copy['cp_type'] = df_copy['cp_type'].apply(lambda x: 1 if x == "ctl_vehicle" else 0)
df_copy['cp_dose'] = df_copy['cp_dose'].apply(lambda x: 1 if x == "D2" else 0)
scaler = RobustScaler()
X = df_copy.values
X = scaler.fit_transform(X)

df_X = pd.DataFrame(X, columns=df_copy.columns)
# New pdf for 9 random features
nrows, ncols = 3, 3

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=[36,24])

for i in range(nrows):
    for j in range(ncols):
        axis = ax[i][j]
        feature = random.randint(0, 875)
        column_name = df_train.columns[feature]
        axis.hist(df_train.iloc[:, feature], density=True, bins=100)
        axis.hist(df_X.iloc[:, feature],density=True, bins=100, alpha=0.5, color='red')
        axis.set_title(f'pdf for {df_train.columns[feature]}', {'fontsize': 30})
        axis.legend(['Before', 'After'])

plt.show()
# Robust Scalar on slightly skewed data
nrows, ncols = 2, 2
fig, ax = plt.subplots(figsize=[24, 18], nrows=nrows, ncols=ncols)
skewed_g = [['g-23', 'g-413'], ['g-307', 'g-238']]
cmap = plt.cm.get_cmap("Set2")
colors = cmap.colors
for i in range(nrows):
    for j in range(ncols):
        axis = ax[i][j]
        axis.hist(df_train[skewed_g[i][j]], bins=100, density=True)
        axis.hist(df_X[skewed_g[i][j]], bins=100, density=True, color='red', alpha=0.5)
        axis.set_title(f'pdf for {skewed_g[i][j]}', {'fontsize': 24})
        axis.set_xlabel("Numerical value in training set", {'fontsize': 16})
        axis.set_ylabel("Probability density", {'fontsize': 16})
plt.show()

correlation_matrix = np.corrcoef(df_copy, df_train_scored, False)
# remove instances where feature is correlated to feature, and target is correlated to target
correlation_features = correlation_matrix[:875,875:] # shape 875 x 206
df_correlation = pd.DataFrame(correlation_features, index=df_features.columns, columns=df_targets.columns)
# plot correlation matrix for 10 features and 10 targets
feature = random.randint(0, 865)
target = random.randint(0, 196)

correlation_features_submatrix = correlation_features[feature:feature + 10, target: target + 10]
fig, ax = plt.subplots(figsize=[10,10])
ax.imshow(correlation_features_submatrix)

# Loop over data dimensions and create text annotations.
for i in range(10):
    for j in range(10):
        text = ax.text(j, i, round(correlation_features_submatrix[i, j], 4),
                       ha="center", va="center", color="w")

ax.set_yticks(range(10))
ax.set_yticklabels([df_features.columns[f] for f in np.arange(feature, feature +11)])
ax.set_xticks(range(10))
ax.set_xticklabels([df_targets.columns[t] for t in np.arange(target, target + 11)])
plt.xticks(rotation=45, ha='right')
plt.show()
features = df_train
g = sns.FacetGrid(features, col='cp_type')
g.map(sns.countplot, 'cp_time')
g  = sns.FacetGrid(features, col="cp_type" )
g.map(sns.countplot , 'cp_dose'  )
plt.show()
g  = sns.FacetGrid(features, col="cp_dose" )
g.map(sns.countplot , 'cp_time'  )
plt.show()
# compute the means of cells and genes for further analysis

features['c_mean'] = features[cells].mean(axis=1)
features['g_mean'] = features[genes].mean(axis=1)
# helper functions

def plt_dist(feature, mean_type, plt_num):
    plt.subplot(plt_num)
    for i in features[feature].unique():
        sns.distplot(features[features[feature]==i][mean_type], label=i, hist=False, kde_kws={"shade": True})
    plt.title(f"{mean_type} based on {feature}")
    plt.legend()

def plt_box(feature, mean_type, plt_num):
    plt.subplot(plt_num)
    sns.boxplot(x=features[feature], y=features[mean_type])
    plt.title(f"{mean_type} based on {feature}")
    plt.legend()
    
fig, axs = plt.subplots(figsize=(16,16), nrows=2, ncols=3)


plt_details = [
        (231, "g_mean", "cp_type"),
        (232, "g_mean", "cp_time"),
        (233, "g_mean", "cp_dose"),    
        (234, "g_mean", "cp_type"),
        (235, "g_mean", "cp_time"),
        (236, "g_mean", "cp_dose"),
        ]


for (plt_num, mean_type, feature) in plt_details[:3]:
    plt_dist(feature, mean_type, plt_num)
    
for (plt_num, mean_type, feature) in plt_details[3:6]:
    plt_box(feature, mean_type, plt_num)
    

plt.show()
fig, axs = plt.subplots(figsize=(16,16), nrows=2, ncols=3)


plt_details = [
        (231, "c_mean", "cp_type"),
        (232, "c_mean", "cp_time"),
        (233, "c_mean", "cp_dose"),    
        (234, "c_mean", "cp_type"),
        (235, "c_mean", "cp_time"),
        (236, "c_mean", "cp_dose"),
        ]


for (plt_num, mean_type, feature) in plt_details[:3]:
    plt_dist(feature, mean_type, plt_num)
    
for (plt_num, mean_type, feature) in plt_details[3:6]:
    plt_box(feature, mean_type, plt_num)
    

plt.show()
target = df_train_scored.drop(['sig_id'] , axis =1)


feat_target  = pd.merge(features , df_train_scored , how = "inner" , on = ['sig_id','sig_id'])
target_cols = list(target)
feat_target["target_sum"] = feat_target[target_cols].sum(axis =1)
feat_target.drop("sig_id" , axis = 1, inplace = True)

fig,ax = plt.subplots(figsize=(16,9))
plt.subplot(131)
sns.countplot(x = 'target_sum' , hue= 'cp_type', data = feat_target)
plt.subplot(132)
sns.countplot(x = 'target_sum' , hue= 'cp_time', data = feat_target)
plt.subplot(133)
sns.countplot(x = 'target_sum' , hue= 'cp_dose', data = feat_target)

plt.show()
fig,ax = plt.subplots(figsize=(16,9))
plt.subplot(121)
sns.barplot(x = 'target_sum' , y= 'c_mean', data = feat_target)
plt.subplot(122)
sns.barplot(x = 'target_sum' , y= 'g_mean', data = feat_target)

plt.show()
corr = features[genes[:99]].corr() # taking only first 99 genes other wise its a mess
f, ax = plt.subplots(figsize=(45, 45))
# Add diverging colormap from red to blue
cmap = sns.diverging_palette(250, 10, as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# plot the heatmap
sns.heatmap(corr,  mask = mask,
        xticklabels=corr.columns,
        yticklabels=corr.columns , cmap=cmap)
plt.show()
corr = features[cells].corr()
f, ax = plt.subplots(figsize=(45, 45))
# Add diverging colormap from red to blue
cmap = sns.diverging_palette(250, 10, as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# plot the heatmap
sns.heatmap(corr,  mask = mask,
        xticklabels=corr.columns,
        yticklabels=corr.columns , cmap=cmap)
plt.show()
corr = target.corr()
f, ax = plt.subplots(figsize=(45, 45))
# Add diverging colormap from red to blue
cmap = sns.diverging_palette(250, 10, as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# plot the heatmap
sns.heatmap(corr,  mask = mask,
        xticklabels=corr.columns,
        yticklabels=corr.columns , cmap=cmap)
plt.show()

kot = corr[corr>=.5]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Reds" )
plt.show()