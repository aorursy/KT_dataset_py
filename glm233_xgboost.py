

#导入、查看文件目录





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd 

import numpy as np



import seaborn as sns

sns.set_style("dark")

import matplotlib.pyplot as plt





from sklearn.manifold import TSNE

from sklearn.decomposition import TruncatedSVD



from time import time



import os



import warnings

warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

df_test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

train_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_non_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
features = df_train

features.info()
#test_features 前五行数据

df_train.head()

# test

df_test.head()
# scored

train_scored.head()
# non scored

train_non_scored.head()
#查看空值

df_train.isnull().sum().sum()
# check For missing values 

df_test.isnull().sum().sum()
#查看整个表格MOA非零占比

scored = train_scored.drop(columns = ["sig_id"] , axis = 1)

print((scored.to_numpy()).sum()/(scored.shape[0]*scored.shape[1])*100 , "%")
non_scored = train_non_scored.drop(columns = ["sig_id"] , axis = 1)

print((non_scored.to_numpy()).sum()/(non_scored.shape[0]*non_scored.shape[1])*100 , "%")
common  = ['sig_id', 'cp_type','cp_time','cp_dose']

genes = list(filter(lambda x : "g-" in x  , list(features)))

cells = list(filter(lambda x : "c-" in x  , list(features)))

#提取基因和细胞列表
plt.figure(figsize=(6,6))

ax = sns.countplot(features["cp_type"] , palette="Set2")

ax.set_title("treatment")





plt.show()
plt.figure(figsize=(6,6))

ax = sns.countplot(features["cp_dose"] , palette="Set2")

ax.set_title("Dose")

plt.show()
plt.figure(figsize=(6,6))

ax = sns.countplot(features["cp_time"] , palette="Set2")

ax.set_title("Time")

plt.show()
fig, axs = plt.subplots(ncols=2 , nrows = 2 , figsize=(9, 9))

sns.distplot(features['g-0'] ,color="b", kde_kws={"shade": True}, ax=axs[0][0] )

sns.distplot(features['g-1'] ,color="r", kde_kws={"shade": True}, ax=axs[0][1] )

sns.distplot(features['g-2'], color="g", kde_kws={"shade": True}, ax=axs[1][0] )

sns.distplot(features['g-3'] ,color="y", kde_kws={"shade": True}, ax=axs[1][1] )

plt.show()
# some stats plot for genes

fig, axs = plt.subplots(ncols=2 , nrows = 2 , figsize=(13,13))

sns.distplot(features[genes].max(axis =1) ,color="b",hist=False, kde_kws={"shade": True}, ax=axs[0][0] ).set(title = 'max')

sns.distplot(features[genes].min(axis =1) ,color="r",hist=False, kde_kws={"shade": True}, ax=axs[0][1] ).set(title = 'min')

sns.distplot(features[genes].mean(axis =1), color="g",hist=False, kde_kws={"shade": True}, ax=axs[1][0] ).set(title = 'mean')

sns.distplot(features[genes].std(axis =1) ,color="y",hist=False, kde_kws={"shade": True}, ax=axs[1][1] ).set(title = 'sd')

plt.show()
fig, axs = plt.subplots(ncols=2 , nrows = 2 , figsize=(9, 9))

sns.distplot(features['c-0'] ,color="b", kde_kws={"shade": True}, ax=axs[0][0] )

sns.distplot(features['c-1'] ,color="r", kde_kws={"shade": True}, ax=axs[0][1] )

sns.distplot(features['c-2'], color="g", kde_kws={"shade": True}, ax=axs[1][0] )

sns.distplot(features['c-3'] ,color="y", kde_kws={"shade": True}, ax=axs[1][1] )

plt.show()
fig, axs = plt.subplots(ncols=2 , nrows = 2 , figsize=(13,13))

sns.distplot(features[cells].max(axis =1) ,color="b",hist=False, kde_kws={"shade": True}, ax=axs[0][0] ).set(title = 'max')

sns.distplot(features[cells].min(axis =1) ,color="r",hist=False, kde_kws={"shade": True}, ax=axs[0][1] ).set(title = 'min')

sns.distplot(features[cells].mean(axis =1), color="g",hist=False, kde_kws={"shade": True}, ax=axs[1][0] ).set(title = 'mean')

sns.distplot(features[cells].std(axis =1) ,color="y",hist=False, kde_kws={"shade": True}, ax=axs[1][1] ).set(title = 'sd')

plt.show()
target  = train_scored.drop(['sig_id'] , axis =1)



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


g  = sns.FacetGrid(features, col="cp_type" )

g.map(sns.countplot , 'cp_time'  )

plt.show()



# sns.countplot(x = features['cp_time']  )

g  = sns.FacetGrid(features, col="cp_type" )

g.map(sns.countplot , 'cp_dose'  )

plt.show()
g  = sns.FacetGrid(features, col="cp_dose" )

g.map(sns.countplot , 'cp_time'  )

plt.show()
# g_mean and  c_mean and g_mean for analysis.

features['c_mean'] = features[cells].mean(axis =1)

features['g_mean'] = features[genes].mean(axis =1)



fig, axs = plt.subplots(figsize=(16,16) , nrows=2 , ncols =3)

plt.subplot(231)

for i in features.cp_type.unique():

    sns.distplot(features[features['cp_type']==i]['g_mean'],label=i, hist=False, kde_kws={"shade": True})

plt.title(f"g_mean based on cp_type")

plt.legend()



plt.subplot(232)

for i in features.cp_time.unique():

    sns.distplot(features[features['cp_time']==i]['g_mean'],label=i, hist=False, kde_kws={"shade": True})

plt.title(f"g_mean based on cp_time")

plt.legend()



plt.subplot(233)

for i in features.cp_dose.unique():

    sns.distplot(features[features['cp_dose']==i]['g_mean'],label=i, hist=False, kde_kws={"shade": True})

plt.title(f"g_mean based on cp_dose")

plt.legend()



plt.subplot(234)

sns.boxplot( x = features['cp_type'] , y = features['g_mean'] )

plt.title(f"g_mean based on cp_type")

plt.legend()



plt.subplot(235)

sns.boxplot( x = features['cp_time'] , y = features['g_mean'] )

plt.title(f"g_mean based on cp_time")

plt.legend()



plt.subplot(236)

sns.boxplot( x = features['cp_dose'] , y = features['g_mean'] )

plt.title(f"g_mean based on cp_dose")

plt.legend()



plt.show()



fig, axs = plt.subplots(figsize=(16,16) , nrows=2 , ncols =3)

plt.subplot(231)

for i in features.cp_type.unique():

    sns.distplot(features[features['cp_type']==i]['c_mean'],label=i, hist=False, kde_kws={"shade": True})

plt.title(f"c_mean based on cp_type")

plt.legend()



plt.subplot(232)

for i in features.cp_time.unique():

    sns.distplot(features[features['cp_time']==i]['c_mean'],label=i, hist=False, kde_kws={"shade": True})

plt.title(f"c_mean based on cp_time")

plt.legend()



plt.subplot(233)

for i in features.cp_dose.unique():

    sns.distplot(features[features['cp_dose']==i]['c_mean'],label=i, hist=False, kde_kws={"shade": True})

plt.title(f"c_mean based on cp_dose")

plt.legend()



plt.subplot(234)

sns.boxplot( x = features['cp_type'] , y = features['c_mean'] )

plt.title(f"c_mean based on cp_type")

plt.legend()



plt.subplot(235)

sns.boxplot( x = features['cp_time'] , y = features['c_mean'] )

plt.title(f"c_mean based on cp_time")

plt.legend()



plt.subplot(236)

sns.boxplot( x = features['cp_dose'] , y = features['c_mean'] )

plt.title(f"c_mean based on cp_dose")

plt.legend()



plt.show()



from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



train = df_train.drop(['c_mean', 'g_mean'] , axis=1)

train['type'] = 'train'

test = df_test

test['type'] = 'test'

X = train.append(test)



# label encode cp_type , cp_dose and cp_time

# X = pd.get_dummies(columns = ['cp_type' , 'cp_dose', 'cp_time'], drop_first =True , data = X)

numeric_cols = genes+cells

X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])
pca_genes = PCA(n_components=5)

pca_gene_data = pca_genes.fit_transform(X[genes])

principal_genes = pd.DataFrame(data = pca_gene_data

             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5'])
principal_genes.head()
print('Explained variation per principal component: {}'.format(pca_genes.explained_variance_ratio_))
fig,ax = plt.subplots(figsize=(9, 9))

sns.barplot(x =['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'], y = pca_genes.explained_variance_ratio_*100  )

sns.lineplot(x =['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'], y = pca_genes.explained_variance_ratio_*100, color ="r")

plt.show()
pca_genes = PCA(n_components=2)

pca_gene_data = pca_genes.fit_transform(X[genes])

inter_pc_gene = pd.DataFrame(data = pca_gene_data

             , columns = ['PC1', 'PC2'])

X['PC1_gene'] = inter_pc_gene['PC1']

X['PC2_gene'] = inter_pc_gene['PC2']
fig, ax = plt.subplots(figsize=(9,16))

plt.subplot(311)

sns.scatterplot(

    x="PC1_gene", y="PC2_gene",

    hue="cp_type",

    style = "cp_type",

    data=X,

    legend="full",

)

plt.subplot(312)

sns.scatterplot(

    x="PC1_gene", y="PC2_gene",

    hue="cp_time",

    style = "cp_time",

    data=X,

    legend="full",

)

plt.subplot(313)

sns.scatterplot(

    x="PC1_gene", y="PC2_gene",

    hue="cp_dose",

    style = "cp_dose",

    data=X,

    legend="full",

)

plt.show()
pca_cell = PCA(n_components=5)

pca_cell_data = pca_cell.fit_transform(X[cells])

principal_cell = pd.DataFrame(data = pca_cell_data

             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5'])
principal_cell.head()
print('Explained variation per principal component: {}'.format(pca_cell.explained_variance_ratio_))
fig,ax = plt.subplots(figsize=(9, 9))

sns.barplot(x =['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'], y = pca_cell.explained_variance_ratio_*100  )

sns.lineplot(x =['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'], y = pca_cell.explained_variance_ratio_*100, color ="r")

plt.show()
pca_cell = PCA(n_components=2)

pca_cell_data = pca_cell.fit_transform(X[cells])

inter_pc_cell = pd.DataFrame(data = pca_cell_data

             , columns = ['PC1', 'PC2'])

X['PC1_cell'] = inter_pc_gene['PC1']

X['PC2_cell'] = inter_pc_gene['PC2']
fig, ax = plt.subplots(figsize=(9,16))

plt.subplot(311)

sns.scatterplot(

    x="PC1_cell", y="PC2_cell",

    hue="cp_type",

    style = "cp_type",

    data=X,

    legend="full",

)

plt.subplot(312)

sns.scatterplot(

    x="PC1_cell", y="PC2_cell",

    hue="cp_time",

    style = "cp_time",

    data=X,

    legend="full",

)

plt.subplot(313)

sns.scatterplot(

    x="PC1_cell", y="PC2_cell",

    hue="cp_dose",

    style = "cp_dose",

    data=X,

    legend="full",

)

plt.show()
X = pd.get_dummies(columns = ['cp_type' , 'cp_dose', 'cp_time'], drop_first =True , data = X) # dummification is important here

X_train = X[X['type']  == 'train'][['PC1_gene', 'PC2_gene', 'PC1_cell', 'PC2_cell', 'cp_type_trt_cp', 'cp_dose_D2', 'cp_time_48', 'cp_time_72']]

Y_train = target

X_test = X[X['type']  == 'test'][['PC1_gene', 'PC2_gene', 'PC1_cell', 'PC2_cell', 'cp_type_trt_cp', 'cp_dose_D2', 'cp_time_48', 'cp_time_72']]
import xgboost as xgb

from sklearn.datasets import make_multilabel_classification

from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import accuracy_score







x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=77)



xgb_estimator = xgb.XGBClassifier(objective='binary:logistic')



multilabel_model = MultiOutputClassifier(xgb_estimator)



multilabel_model.fit(x_train, y_train)

preds = multilabel_model.predict(x_test)

# evaluate on test data

print('Accuracy on test data: {:.1f}%'.format(accuracy_score(y_test,preds )*100))
preds = multilabel_model.predict(X_test)
df= pd.DataFrame(preds , columns = list(target) , index =df_test['sig_id']  )
df.to_csv('submission.csv')