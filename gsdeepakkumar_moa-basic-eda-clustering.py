# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from collections import Counter

from scipy import stats

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv("../input/lish-moa/train_targets_scored.csv")

df_train_nonscored=pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")

train_features=pd.read_csv("../input/lish-moa/train_features.csv")

test_features=pd.read_csv("../input/lish-moa/test_features.csv")
df_train.shape,df_train_nonscored.shape,train_features.shape,test_features.shape
train_features.head()
train_features['cp_type'].value_counts()
train_features['cp_time'].value_counts()
train_features['cp_dose'].value_counts()
## Map the categorical features,



train_features['cp_type']=train_features['cp_type'].map({'trt_cp':'0','ctl_vehicle':'1'})

train_features['cp_time']=train_features['cp_time'].map({48:'0',72:'1',24:'2'})

train_features['cp_dose']=train_features['cp_dose'].map({'D1':'0','D2':'1'})
gene_expressions=train_features.columns.str.startswith('g-')

sum(gene_expressions)
plt.figure(figsize=(15,15))

plt.subplot(2,2,1)

sns.distplot(train_features.loc[:,gene_expressions].median())

plt.title("Distribution of Gene Expression - Plot of Median",fontsize=16)

plt.xlabel("Median of gene expression",fontsize=8)

plt.subplot(2,2,2)

sns.distplot(train_features.loc[:,gene_expressions].max())

plt.title("Distribution of Gene Expression - Plot of Max",fontsize=16)

plt.xlabel("Max of gene expression",fontsize=8)

plt.subplot(2,2,3)

sns.distplot(train_features.loc[:,gene_expressions].min())

plt.title("Distribution of Gene Expression - Plot of Min",fontsize=16)

plt.xlabel("Min of gene expression",fontsize=8)

plt.subplot(2,2,4)

sns.distplot(train_features.loc[:,gene_expressions].std())

plt.title("Distribution of Gene Expression - Plot of std",fontsize=16)

plt.xlabel("Std of gene expression",fontsize=8)
##Reference -Chris Delotte's kernel.

plt.figure(figsize=(15,15))

plt.subplot(3,3,9)

for i in range(9):

    plt.subplot(3,3,i+1)

    sns.distplot(train_features.iloc[:,i+4])

    plt.title(train_features.columns[i+4])

    plt.xlabel('')



plt.subplots_adjust(hspace=0.4)

plt.show()



train_dur_48=train_features[train_features['cp_time']=='0']

fig=plt.figure(figsize=(10,10))

plt.subplot(3,3,9)

for i in range(9):

    plt.subplot(3,3,i+1)

    sns.distplot(train_dur_48.iloc[:,i+4])

    plt.title(train_dur_48.columns[i+4])

fig.suptitle("For treatment duration 48 hrs Gene expression features")

fig.tight_layout()

fig.subplots_adjust(top=0.88)

plt.show()
train_dur_72=train_features[train_features['cp_time']=='1']

fig=plt.figure(figsize=(10,10))

plt.subplot(3,3,9)

for i in range(9):

    plt.subplot(3,3,i+1)

    sns.distplot(train_dur_72.iloc[:,i+4])

    plt.title(train_dur_72.columns[i+4])

fig.suptitle("For treatment duration 72 hrs gene expression features")

fig.tight_layout()

fig.subplots_adjust(top=0.88)

plt.show()
train_dur_24=train_features[train_features['cp_time']=='2']

fig=plt.figure(figsize=(10,10))

plt.subplot(3,3,9)

for i in range(9):

    plt.subplot(3,3,i+1)

    sns.distplot(train_dur_24.iloc[:,i+4])

    plt.title(train_dur_24.columns[i+4])

fig.suptitle("For treatment duration 24 hrs gene expression features")

fig.tight_layout()

fig.subplots_adjust(top=0.88)

plt.show()
cell_viability=train_features.columns.str.startswith('c-')

sum(cell_viability)
plt.figure(figsize=(15,15))

plt.subplot(2,2,1)

sns.distplot(train_features.loc[:,cell_viability].median())

plt.title("Distribution of Cell Viability - Plot of Median",fontsize=16)

plt.xlabel("Median of Cell Viability",fontsize=8)

plt.subplot(2,2,2)

sns.distplot(train_features.loc[:,cell_viability].max())

plt.title("Distribution of Cell Viability - Plot of Max",fontsize=16)

plt.xlabel("Max of Cell Viability",fontsize=8)

#plt.subplot(2,2,3)

# sns.distplot(train_features.loc[:,cell_viability].min())

# plt.title("Distribution of Cell Viability - Plot of Min",fontsize=16)

# plt.xlabel("Min of Cell Viability",fontsize=8)

plt.subplot(2,2,3)

sns.distplot(train_features.loc[:,cell_viability].std())

plt.title("Distribution of Cell Viability - Plot of std",fontsize=16)

plt.xlabel("Std of Cell Viability",fontsize=8)
train_features.loc[:,cell_viability].min().value_counts()
##Reference -Chris Delotte's kernel.

plt.figure(figsize=(15,15))

plt.subplot(3,3,9)

for i in range(9):

    plt.subplot(3,3,i+1)

    sns.distplot(train_features.iloc[:,i+776])

    plt.title(train_features.columns[i+776])



plt.subplots_adjust(hspace=0.4)

plt.show()
fig=plt.figure(figsize=(10,10))

plt.subplot(3,3,9)

for i in range(9):

    plt.subplot(3,3,i+1)

    sns.distplot(train_dur_48.iloc[:,i+776])

    plt.title(train_dur_48.columns[i+776])

fig.suptitle("For treatment duration 48 hrs Cell Viability features")

fig.tight_layout()

fig.subplots_adjust(top=0.88)

plt.show()
fig=plt.figure(figsize=(10,10))

plt.subplot(3,3,9)

for i in range(9):

    plt.subplot(3,3,i+1)

    sns.distplot(train_dur_72.iloc[:,i+776])

    plt.title(train_dur_72.columns[i+776])

fig.suptitle("For treatment duration 72 hrs Cell Viability features")

fig.tight_layout()

fig.subplots_adjust(top=0.88)

plt.show()
fig=plt.figure(figsize=(10,10))

plt.subplot(3,3,9)

for i in range(9):

    plt.subplot(3,3,i+1)

    sns.distplot(train_dur_24.iloc[:,i+776])

    plt.title(train_dur_24.columns[i+776])

fig.suptitle("For treatment duration 24 hrs Cell Viability features")

fig.tight_layout()

fig.subplots_adjust(top=0.88)

plt.show()
feats=[col for col in train_features.columns if col not in ['sig_id']]

len(feats)
##Correlation between the features:

#Reference https://www.kaggle.com/senkin13/eda-starter#Feature-Correlation

correlations=train_features[feats].corr().abs().unstack().sort_values(kind='quicksort').reset_index()

correlations=correlations[correlations['level_0']!=correlations['level_1']]

correlations.head(10)
correlations.tail(10)
df_train.head()
target_sums=df_train.iloc[:,1:].sum(axis=0)
## Agents class names,

print('Total Agent classes',len(target_sums.loc[target_sums.index.str.contains('_agent')]))

print('Agent class names & total active MOA for each class')

print(target_sums.loc[target_sums.index.str.contains('_agent')])
## Receptors class names,

print('Total Receptors',len(target_sums.loc[target_sums.index.str.contains('_agonist')])+len(target_sums.loc[target_sums.index.str.contains('_antagonist')]))

# print('Agonist class names & total active MOA for each class')

# print(target_sums.loc[target_sums.index.str.contains('_agonist')])

# print('Antagonist class names & total active MOA for each class')

# print(target_sums.loc[target_sums.index.str.contains('_antagonist')])
## Enzymes class names,

print('Total Enzymes class names',len(target_sums.loc[target_sums.index.str.contains('_inhibitor')])+len(target_sums.loc[target_sums.index.str.contains('_activator')])+len(target_sums.loc[target_sums.index.str.contains('_blocker')]))
# print('Inhibitor class names & total active MOA for each class')

# print(target_sums.loc[target_sums.index.str.contains('_inhibitor')])

# print('Activator class names & total active MOA for each class')

# print(target_sums.loc[target_sums.index.str.contains('_activator')])

# print('Blocker class names & total active MOA for each class')

# print(target_sums.loc[target_sums.index.str.contains('_blocker')])
#Other class names,

print("Total Other class names",len(target_sums.loc[~((target_sums.index.str.contains('_agent')) | 

                  (target_sums.index.str.contains('_inhibitor')) |

                  (target_sums.index.str.contains('_agonist')) | 

                  (target_sums.index.str.contains('_antagonist') | 

                   (target_sums.index.str.contains('_activator') |

                    (target_sums.index.str.contains('_blocker')))))]))

print(target_sums.loc[~((target_sums.index.str.contains('_agent')) | 

                  (target_sums.index.str.contains('_inhibitor')) |

                  (target_sums.index.str.contains('_agonist')) | 

                  (target_sums.index.str.contains('_antagonist') | 

                   (target_sums.index.str.contains('_activator') |

                    (target_sums.index.str.contains('_blocker')))))])
print(f'''Percentage of inhibitors present {(target_sums[target_sums.index.str.contains('_inhibitor')].count()/206)*100:.2f} %''')

print(f'''Percentage of antagonist present {(target_sums[target_sums.index.str.contains('_antagonist')].count()/206)*100:.2f} %''')

print(f'''Percentage of agonist present {(target_sums[target_sums.index.str.contains('_agonist')].count()/206)*100:.2f} %''')
#Counter([i[-1] for i in target_sums.index.str.split("_")])
df_train.loc[:,target_sums.sort_values(ascending=False).head(5).index].sum(axis=0)
df_train.loc[:,target_sums.sort_values(ascending=False).tail(5).index].sum(axis=0)
df_train.loc[:,target_sums.sort_values(ascending=False).tail(10).index].sum(axis=0)
plt.figure(figsize=(8,8))

sns.countplot(df_train.iloc[:,1:].sum(axis=1))

plt.title("Distribution of Multi-labels in target",fontsize=16)

plt.xlabel("Label Count",fontsize=8)
target_stat=df_train.iloc[:,1:].sum(axis=1).reset_index(drop=True)

(target_stat.value_counts()/target_stat.shape[0])*100
target_stat.value_counts()
single_moa=df_train.loc[(df_train.iloc[:,1:].sum(axis=1))==1]
(single_moa.iloc[:,1:].idxmax(axis=1).value_counts()/12532)*100
len(df_train.loc[(df_train.iloc[:,1:].sum(axis=1))==1])
df_train_nonscored.head()
plt.figure(figsize=(8,8))

sns.countplot(df_train_nonscored.iloc[:,1:].sum(axis=1))

plt.title("Distribution of Multi-labels in target(in Non Scored Data)",fontsize=16)

plt.xlabel("Label Count",fontsize=8)
target_nonsco_stat=df_train_nonscored.iloc[:,1:].sum(axis=1).reset_index(drop=True)

(target_nonsco_stat.value_counts()/target_nonsco_stat.shape[0])*100
set(df_train_nonscored['sig_id'])-set(df_train['sig_id'])
len(set(df_train_nonscored.columns)-set(df_train.columns))
target_sum_nonscored=df_train_nonscored.iloc[:,1:].sum(axis=0)
set([i[-1] for i in  target_sum_nonscored.index.str.split('_')])
from sklearn import metrics

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

from sklearn.metrics import silhouette_score

from sklearn.cluster import DBSCAN
gene_expression=train_features.loc[:,train_features.columns.str.contains('g-')]
# distortions = []

# K = range(2, 50)

# for k in K:

#     k_means = KMeans(n_clusters=k, random_state=42).fit(gene_expression)

#     k_means.fit(gene_expression)

#     distortions.append(sum(np.min(cdist(gene_expression, k_means.cluster_centers_, 'euclidean'), axis=1)) / gene_expression.shape[0])
# X_line = [K[0], K[-1]]

# Y_line = [distortions[0], distortions[-1]]



# # Plot the elbow

# plt.plot(K, distortions, 'b-')

# plt.plot(X_line, Y_line, 'r')

# plt.xlabel('k')

# plt.ylabel('Distortion')

# plt.title('The Elbow Method showing the optimal k')

# plt.show()
k=9

k_means=KMeans(n_clusters=k,random_state=42)

clust=k_means.fit_transform(gene_expression)
train_feat_cluster=train_features.copy()

train_feat_cluster['cluster']=k_means.labels_
train_feat_cluster['cluster'].value_counts()
## Visualizing the clusters:

tsne=TSNE(n_components=2,verbose=1,perplexity=100,random_state=42)

train_feat_embedded=tsne.fit_transform(gene_expression.values)
plt.figure(figsize=(10,10))

palette=sns.hls_palette(9,l=0.5,s=0.6)

sns.scatterplot(train_feat_embedded[:,0],train_feat_embedded[:,1],hue=train_feat_cluster['cluster'],palette=palette)

plt.title("t-SNE with cluster",fontsize=10)

plt.xlabel('x',fontsize=8)

plt.ylabel('y',fontsize=8)
