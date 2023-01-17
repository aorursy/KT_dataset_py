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
#quality check

#1.  check the datsets

df_train.head()

# test

df_test.head()
# scored

train_scored.head()
# non scored

train_non_scored.head()
# check For missing values 

df_train.isnull().sum().sum()
# check For missing values 

df_test.isnull().sum().sum()
# check for target sparsity

scored = train_scored.drop(columns = ["sig_id"] , axis = 1)

# non zero target varaibles

print((scored.to_numpy()).sum()/(scored.shape[0]*scored.shape[1])*100 , "%")
non_scored = train_non_scored.drop(columns = ["sig_id"] , axis = 1)

# non zero target_nonscored varaibles

print((non_scored.to_numpy()).sum()/(non_scored.shape[0]*non_scored.shape[1])*100 , "%")
# list the columns 

# list(features)

# get all the gene features and cell features

common  = ['sig_id',

 'cp_type',

 'cp_time',

 'cp_dose']





genes = list(filter(lambda x : "g-" in x  , list(features)))



cells = list(filter(lambda x : "c-" in x  , list(features)))





plt.figure(figsize=(6,6))

ax = sns.countplot(features["cp_type"] , palette="Set2")

ax.set_title("Treatment Type")





plt.show()
plt.figure(figsize=(6,6))

ax = sns.countplot(features["cp_dose"] , palette="Set2")

ax.set_title("Treatment Dose")





plt.show()
plt.figure(figsize=(6,6))

ax = sns.countplot(features["cp_time"] , palette="Set2")

ax.set_title("Treatment time")





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



feat_target  = pd.merge(features , train_scored , how = "inner" , on = ['sig_id','sig_id'])

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
# pca analysis for genes 

# a bit of data cleaning 

# get a train test consolidated DF

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



train = df_train.drop(['c_mean', 'g_mean'] , axis=1)

train['type'] = 'train'

test = df_test

test['type'] = 'test'

X = train.append(test)



# lets label encode cp_type , cp_dose and cp_time

# X = pd.get_dummies(columns = ['cp_type' , 'cp_dose', 'cp_time'], drop_first =True , data = X)

numeric_cols = genes+cells

X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])
# TRY PCA distribution

# from sklearn.decomposition import SparsePCA



# from sklearn.decomposition import TruncatedSVD



# n_comp = [2, 4,8,10,15,20 ,30, 50,100,150,200,300,450,550,700] # list containing different values of components

# explained = [] # explained variance ratio for each component of Truncated SVD

# for x in n_comp:

#     svd = TruncatedSVD(n_components=x)

#     svd.fit(X[genes])

#     explained.append(svd.explained_variance_ratio_.sum())

#     print("Number of components = %r and explained variance = %r"%(x,svd.explained_variance_ratio_.sum()))

# plt.plot(n_comp, explained)

# plt.xlabel('Number of components')

# plt.ylabel("Explained Variance")

# plt.title("Plot of Number of components v/s explained variance")

# plt.show()







# # pca_genes = SparsePCA(n_components=2)

# # # pca_genes = PCA(n_components=5)

# # pca_gene_data = pca_genes.fit_transform(X[genes])

# # principal_genes = pd.DataFrame(data = pca_gene_data

# #              , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6','principal component 7','principal component 8','principal component 9', 'principal component 10'])
# pca_gene_data

# len(genes)
# print('Explained variation per principal component: {}'.format(pca_genes.explained_variance_ratio_))
# fig,ax = plt.subplots(figsize=(9, 9))

# sns.barplot(x =['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10'], y = pca_genes.explained_variance_ratio_*100  )

# sns.lineplot(x =['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10'], y = pca_genes.explained_variance_ratio_*100, color ="r")

# plt.show()
pca_genes = PCA(n_components=5)

pca_gene_data = pca_genes.fit_transform(X[genes])

principal_genes = pd.DataFrame(data = pca_gene_data

             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5'])
print('Explained variation per principal component: {}'.format(pca_genes.explained_variance_ratio_))
fig,ax = plt.subplots(figsize=(9, 9))

sns.barplot(x =['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'], y = pca_genes.explained_variance_ratio_*100  )

sns.lineplot(x =['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'], y = pca_genes.explained_variance_ratio_*100, color ="r")

plt.show()
pca_gene = PCA(n_components=2)

pca_gene_data = pca_gene.fit_transform(X[genes])

inter_pc_gene = pd.DataFrame(data = pca_gene_data

             , columns = ['PC1', 'PC2'])

X['PC1_gene'] = inter_pc_gene['PC1']

X['PC2_gene'] = inter_pc_gene['PC2']
# improve on explained variance

# pca_genes = PCA(n_components=450)

# pca_gene_data = pca_genes.fit_transform(X[genes])

# inter_pc_gene = pd.DataFrame(data = pca_gene_data)

# transformed_genes = [str(i)+"_gene" for i in list(inter_pc_gene)]

# X[transformed_genes]= inter_pc_gene[:]



# X['PC1_gene'] = inter_pc_gene['PC1']

# X['PC2_gene'] = inter_pc_gene['PC2']
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

X['PC1_cell'] = inter_pc_cell['PC1']

X['PC2_cell'] = inter_pc_cell['PC2']
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

# features_final = transformed_genes + ['PC1_cell', 'PC2_cell', 'cp_type_trt_cp', 'cp_dose_D2', 'cp_time_48', 'cp_time_72']

features_final = ['PC1_gene', 'PC2_gene','PC1_cell', 'PC2_cell', 'cp_type_trt_cp', 'cp_dose_D2', 'cp_time_48', 'cp_time_72']
X_train = X[X['type']  == 'train'][features_final]

Y_train = target

X_test = X[X['type']  == 'test'][features_final]
# import xgboost as xgb

# from sklearn.datasets import make_multilabel_classification

# from sklearn.model_selection import train_test_split

# from sklearn.multioutput import MultiOutputClassifier

# from sklearn.metrics import accuracy_score







# # split dataset into training and test set

# x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=77)



# # create XGBoost instance with default hyper-parameters

# xgb_estimator = xgb.XGBClassifier(objective='binary:logistic')



# # create MultiOutputClassifier instance with XGBoost model inside

# multilabel_model = MultiOutputClassifier(xgb_estimator)



# # fit the model

# multilabel_model.fit(x_train, y_train)

# preds = multilabel_model.predict(x_test)

# # evaluate on test data

# print('Accuracy on test data: {:.1f}%'.format(accuracy_score(y_test,preds )*100))
# preds = multilabel_model.predict(X_test)
# df= pd.DataFrame(preds , columns = list(target) , index =df_test['sig_id']  )
# df.to_csv('submission.csv')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from category_encoders import CountEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss



import matplotlib.pyplot as plt



from sklearn.multioutput import MultiOutputClassifier



import os

import warnings

warnings.filterwarnings('ignore')
x = X_train.to_numpy()

y = Y_train.to_numpy()

x_test = X_test.to_numpy()
classifier = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))

# classifier = MultiOutputClassifier(XGBClassifier())



clf = Pipeline([

                ('classify', classifier)

               ])

#  params = {'classify__estimator__colsample_bytree': 0.6522,

#           'classify__estimator__gamma': 1.6975,

#           'classify__estimator__learning_rate': 0.0103,

#           'classify__estimator__max_delta_step': 1.0706,

#           'classify__estimator__max_depth': 50 ,

#           'classify__estimator__min_child_weight': 9.5800,

#           'classify__estimator__n_estimators': 300,

#           'classify__estimator__subsample': 0.8639

#          }





params = {'classify__estimator__colsample_bytree': 0.6522,

          'classify__estimator__gamma': 3.6975,

          'classify__estimator__learning_rate': 0.0503,

          'classify__estimator__max_delta_step': 2.0706,

          'classify__estimator__max_depth': 10,

          'classify__estimator__min_child_weight': 31.5800,

          'classify__estimator__n_estimators': 166,

          'classify__estimator__subsample': 0.8639

         }



_ = clf.set_params(**params)
SEED = 42

NFOLDS = 10 #increase folds

DATA_DIR = '/kaggle/input/lish-moa/'

np.random.seed(SEED)
oof_preds = np.zeros(y.shape)

test_preds = np.zeros((test.shape[0], y.shape[1]))

oof_losses = []

kf = KFold(n_splits=NFOLDS)

for fn, (trn_idx, val_idx) in enumerate(kf.split(x, y)):

    print('Starting fold: ', fn)

    X_train, X_val = x[trn_idx], x[val_idx]

    y_train, y_val = y[trn_idx], y[val_idx]

    

    # drop where cp_type==ctl_vehicle (baseline)

    ctl_mask = X_train[:,-4]==0

    X_train = X_train[~ctl_mask,:]

    y_train = y_train[~ctl_mask]

    

    clf.fit(X_train, y_train)

    val_preds = clf.predict_proba(X_val) # list of preds per class

    val_preds = np.array(val_preds)[:,:,1].T # take the positive class

    oof_preds[val_idx] = val_preds

    

    loss = log_loss(np.ravel(y_val), np.ravel(val_preds))

    oof_losses.append(loss)

    preds = clf.predict_proba(x_test)

    preds = np.array(preds)[:,:,1].T # take the positive class

    test_preds += preds / NFOLDS

    

print(oof_losses)

print('Mean OOF loss across folds', np.mean(oof_losses))

print('STD OOF loss across folds', np.std(oof_losses))
# set control train preds to 0

control_mask = X[X['type'] =='train']['cp_type_trt_cp'] ==0

oof_preds[control_mask] = 0



print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))
# # set control train preds to 0

# control_mask = X[X['type'] =='train']['cp_type_trt_cp'] ==0

# oof_preds[control_mask] = 0



# print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))
# set control test preds to 0

control_mask = X[X['type'] =='test']['cp_type_trt_cp'] == 0

test_preds[control_mask] = 0
# create the submission file

sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')

sub.iloc[:,1:] = test_preds

sub.to_csv('submission.csv', index=False)
sub.head()
# from keras.models import Sequential

# from keras.layers import Dense

# from keras.optimizers import SGD

# from keras.metrics import binary_accuracy

# from keras.utils import to_categorical

# from sklearn.metrics import accuracy_score , log_loss



# clf = Sequential()

# clf.add(Dense(5, activation='relu', input_dim=8))

# clf.add(Dense(206, activation='sigmoid'))

# clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=[binary_accuracy])

# clf.fit(x_train, y_train, epochs=100, batch_size=100, verbose=0)
# preds = clf.predict(x_test)
# from sklearn.metrics import accuracy_score

# accuracy_score(y_test,preds.round())
# print(log_loss(y_test,preds.round()))