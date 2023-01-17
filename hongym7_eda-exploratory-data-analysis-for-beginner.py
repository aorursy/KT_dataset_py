import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
color = sns.color_palette()

from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing as pp 
from scipy.stats import pearsonr 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss 
from sklearn.metrics import precision_recall_curve, average_precision_score 
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report 
DATA_DIR  = os.path.join('/kaggle/input/lish-moa')
TRAIN_FEATURE_FILE = os.path.join(DATA_DIR, 'train_features.csv')
TEST_FEATURE_FILE = os.path.join(DATA_DIR, 'test_features.csv')

TRAIN_TRAGET_FILE = os.path.join(DATA_DIR, 'train_targets_scored.csv')
SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')
train_feat = pd.read_csv(TRAIN_FEATURE_FILE)
test_feat = pd.read_csv(TEST_FEATURE_FILE)
train_feat.head()
train_feat.describe()
train_target = pd.read_csv(TRAIN_TRAGET_FILE)
train_target.head()
train_target.describe()
y_columns = train_target.drop(columns='sig_id', axis=0)
y_columns.columns
train_target_2 = pd.DataFrame(train_target['11-beta-hsd1_inhibitor'].value_counts())
train_target_2.reset_index(inplace=True)
train_target_2.columns = ['value','count']

print(train_target_2)

plt.figure(figsize = (5, 5))
plt.title('11-beta-hsd1_inhibitor')
g = sns.barplot(x="value", y="count", data=train_target_2, palette="pastel")


plt.show()
train = pd.merge(train_feat, train_target, on='sig_id')
train_filter = train[train['5-alpha_reductase_inhibitor'] == 1]
train_filter = train_filter.iloc[:, :876]
train_filter
corr = train_filter.corr(method = 'pearson')
corr
#df_heatmap = sns.heatmap(corr, cbar = True, annot = True, annot_kws={'size' : 20}, fmt = '.2f', square = True, cmap = 'Blues')
train_feat = train_feat.drop(columns=['sig_id'], axis=0)
test_feat = test_feat.drop(columns=['sig_id'], axis=0)

for feature in ['cp_type', 'cp_dose', 'cp_time']:
    le = LabelEncoder()
    le.fit(list(train_feat[feature].astype(str).values) + list(test_feat[feature].astype(str).values))
    train_feat[feature] = le.transform(list(train_feat[feature].astype(str).values))
    test_feat[feature] = le.transform(list(test_feat[feature].astype(str).values))

print(train_feat.head(10))

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()

fitted = std_scaler.fit(train_feat)

train_feat_scale = std_scaler.transform(train_feat)
train_feat_scale = pd.DataFrame(train_feat_scale, columns=train_feat.columns, index=list(train_feat.index.values))

print(train_feat_scale.head(10))
from sklearn.decomposition import PCA

train_index = range(0, len(train_feat_scale))

n_components = 872
whiten = False
random_state = 2020

pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
X_train_PCA = pca.fit_transform(train_feat_scale)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)
print("Variance Explained by all 872 principal components: ", sum(pca.explained_variance_ratio_))
importanceOfPrincipalComponents = pd.DataFrame(data=pca.explained_variance_ratio_)
importanceOfPrincipalComponentsT =importanceOfPrincipalComponents.T

print('Variance Captured By First 10 Pricipal Components: ',
     importanceOfPrincipalComponentsT.loc[:, 0:9].sum(axis=1).values)
print('Variance Captured By First 20 Pricipal Components: ',
     importanceOfPrincipalComponentsT.loc[:, 0:19].sum(axis=1).values)
print('Variance Captured By First 100 Pricipal Components: ',
     importanceOfPrincipalComponentsT.loc[:, 0:99].sum(axis=1).values)
print('Variance Captured By First 200 Pricipal Components: ',
     importanceOfPrincipalComponentsT.loc[:, 0:199].sum(axis=1).values)
print('Variance Captured By First 300 Pricipal Components: ',
     importanceOfPrincipalComponentsT.loc[:, 0:299].sum(axis=1).values)
print('Variance Captured By First 400 Pricipal Components: ',
     importanceOfPrincipalComponentsT.loc[:, 0:399].sum(axis=1).values)
print('Variance Captured By First 500 Pricipal Components: ',
     importanceOfPrincipalComponentsT.loc[:, 0:499].sum(axis=1).values)
importanceOfPrincipalComponentsT
sns.set(rc={'figure.figsize':(10,10)})
sns.barplot(data=importanceOfPrincipalComponentsT.loc[:,0:9], palette="pastel")
#temp = train_target.iloc[:, 5:20]
#temp = temp[(temp['acetylcholinesterase_inhibitor'] == 1) | (temp['adenosine_receptor_agonist'] == 1) | (temp['adenylyl_cyclase_activator'] == 1)]
#temp.head(20)
def scatterPlot(xDF, yDF, algoName):
    tempDF = pd.DataFrame(data=xDF.loc[:, 0:1], index=xDF.index)
    tempDF = pd.concat((tempDF, yDF), axis=1, join='inner')
    tempDF.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x='First Vector', y='Second Vector', hue='Label', data=tempDF, fit_reg=False)
    
    ax = plt.gca()
    ax.set_title('Target Value  :  ' + algoName + '  ' + str(np.sum(tempDF['Label'])))   
    
    #print(np.sum(tempDF['Label']))

    
def scatterPlot2(xDF, yDF, algoName, column1, column2):
    tempDF = pd.DataFrame(data=xDF.loc[:, [column1, column2]], index=xDF.index)
    tempDF = pd.concat((tempDF, yDF), axis=1, join='inner')
    tempDF.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x='First Vector', y='Second Vector', hue='Label', data=tempDF, fit_reg=False)
    
    ax = plt.gca()
    ax.set_title('Separation of Observations using ' + algoName)
for col in y_columns.columns[0:10]:
    scatterPlot(X_train_PCA, train_target[col], col)

scatterPlot2(train_feat_scale, train_target['trpv_agonist'], 'PCA', 'g-0', 'cp_time')
column_sum = np.sum(train_target, axis=0).to_frame()
column_sum.reset_index(inplace=True)

column_sum.columns = ['y_name', 'count']

column_sum = column_sum.iloc[1:,:]
column_sum = column_sum.sort_values(by=['count'], axis=0, ascending=False)

column_sum_top20 = column_sum[:20]

print(column_sum_top20)

plt.figure(figsize = (20, 10))
plt.title('# of true labels per column')
g = sns.barplot(x=column_sum_top20['y_name'], y=column_sum_top20['count'], data=column_sum_top20, palette="pastel")
g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.show()

column_sum_bottom20 = column_sum[-20:]

print(column_sum_bottom20)

plt.figure(figsize = (20, 10))
plt.title('# of true labels per column')
g = sns.barplot(x=column_sum_bottom20['y_name'], y=column_sum_bottom20['count'], data=column_sum_bottom20, palette="pastel")
g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.show()
row_sum = np.sum(train_target, axis=1)

row_sum_vc = pd.DataFrame(row_sum.value_counts())
row_sum_vc.reset_index(inplace=True)
row_sum_vc.columns = ['tlc','count']

print(row_sum_vc)

plt.figure(figsize = (10, 10))
plt.title('# of true labels per row')
sns.barplot(x=row_sum_vc['tlc'], y=row_sum_vc['count'], data=row_sum_vc, palette="pastel")
plt.xlabel('target label count')

plt.show()
ratio_0_1 = row_sum_vc[(row_sum_vc['tlc'] == 0) | (row_sum_vc['tlc'] == 1)]['count'].sum() / row_sum_vc['count'].sum()
print(f'As for the number of targets for each row, 0 and 1 occupy {ratio_0_1} percent.')
train_columns = train_feat_scale.columns.to_list()
g_list = [i for i in train_columns if i.startswith('g-')]
c_list = [i for i in train_columns if i.startswith('c-')]
train_feat_g = train_feat_scale[g_list]
train_feat_c = train_feat_scale[c_list]
train_feat_g
train_feat_c
train_index = range(0, len(train_feat_g))

n_components = 772
whiten = False
random_state = 2020

pca_g_feat = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

X_train_PCA_g = pca_g_feat.fit_transform(train_feat_g)
X_train_PCA_g = pd.DataFrame(data=X_train_PCA_g, index=train_index)

importanceOfPrincipalComponents = pd.DataFrame(data=pca_g_feat.explained_variance_ratio_)
importanceOfPrincipalComponentsT =importanceOfPrincipalComponents.T

print(importanceOfPrincipalComponents)
#sns.set(rc={'figure.figsize':(10,10)})
sns.barplot(data=importanceOfPrincipalComponentsT.loc[:,0:9], palette="pastel")
train_index = range(0, len(train_feat_c))

n_components = 100
whiten = False
random_state = 2020

pca_c_feat = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

X_train_PCA_c = pca_c_feat.fit_transform(train_feat_c)
X_train_PCA_c = pd.DataFrame(data=X_train_PCA_c, index=train_index)

importanceOfPrincipalComponents = pd.DataFrame(data=pca_c_feat.explained_variance_ratio_)
importanceOfPrincipalComponentsT =importanceOfPrincipalComponents.T

print(importanceOfPrincipalComponents)
#sns.set(rc={'figure.figsize':(10,10)})
sns.barplot(data=importanceOfPrincipalComponentsT.loc[:,0:9], palette="pastel")
