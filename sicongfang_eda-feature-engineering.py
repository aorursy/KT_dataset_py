# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

rcParams = plt.rcParams.copy()

import seaborn as sns

plt.rcParams = rcParams

% matplotlib inline

plt.rcParams["figure.dpi"] = 100

np.set_printoptions(precision=3, suppress=True)

plt.style.use(['fivethirtyeight'])
# Load data to dataframe

raw_train = pd.read_csv("../input/train.csv")

raw_test = pd.read_csv("../input/test.csv")

raw_train.head(50)

raw_test.head(50)
# Create training and test data

X_train = raw_train.drop(columns=["ID_code","target"])

y_train = raw_train[["target"]] 

X_test = raw_test.drop(columns=["ID_code"])

y_test = raw_test[["ID_code"]]
X_train.head()
y_train.head(20)
# Check X training data dimensions

X_train.shape
X_train.describe()
# Check y training data distribution

sns.countplot(x="target", data=y_train)

print(y_train['target'].value_counts()/y_train.shape[0])

print('{} samples are positive'.format(np.sum(y_train['target'] == 1)))

print('{} samples are negative'.format(np.sum(y_train['target'] == 0)))
df1 = pd.concat([X_train.apply(lambda x: sum(x.isnull())).rename("num_missing"),

                 X_train.apply(lambda x: sum(x==0)).rename("num_zero"),

                 X_train.apply(lambda x: len(np.unique(x))).rename("num_unique")],axis=1).sort_values(by=['num_unique'])

df1
df2 = pd.concat([X_test.apply(lambda x: sum(x.isnull())).rename("num_missing"),

                 X_test.apply(lambda x: sum(x==0)).rename("num_zero"),

                 X_test.apply(lambda x: len(np.unique(x))).rename("num_unique")],axis=1).sort_values(by=['num_unique'])

df2
# No missing value comfirmed

np.sum(df1['num_missing']!=0)
sns.distplot(a=X_train['var_71'],rug=True)
sns.distplot(a=X_train['var_131'],rug=True)
#create a function which makes the plot:

from matplotlib.ticker import FormatStrFormatter

def visualize_numeric(ax1, ax2, ax3, df, col, target):

    #plot histogram:

    df.hist(column=col,ax=ax1,bins=200)

    ax1.set_xlabel('Histogram')

    

    #plot box-whiskers:

    df.boxplot(column=col,by=target,ax=ax2)

    ax2.set_xlabel('Transactions')

    

    #plot top 10 counts:

    cnt = df[col].value_counts().sort_values(ascending=False)

    cnt.head(10).plot(kind='barh',ax=ax3)

    ax3.invert_yaxis()  # labels read top-to-bottom

#     ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) #somehow not working 

    ax3.set_xlabel('Count')
for col in list(df1.index[:20]):

    fig, axes = plt.subplots(1, 3,figsize=(10,3))

    ax11 = plt.subplot(1, 3, 1)

    ax21 = plt.subplot(1, 3, 2)

    ax31 = plt.subplot(1, 3, 3)

    fig.suptitle('Feature: %s'%col,fontsize=5)

    visualize_numeric(ax11,ax21,ax31,raw_train,col,'target')

    plt.tight_layout()
#get vars except target:

x_vars = X_train.columns
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA

from sklearn.preprocessing import RobustScaler



pca = make_pipeline(RobustScaler(), PCA(n_components=2))

train_pca = pca.fit_transform(X_train[x_vars])

plt.scatter(train_pca[:, 0], train_pca[:, 1], c=y_train['target'], alpha=.1)

plt.xlabel("first principal component")

plt.ylabel("second principal component")
from sklearn.pipeline import Pipeline

pca_50 = PCA(n_components=50)

# pipe = Pipeline(steps=[('sampler', RobustScaler()),('pca', pca_50)])

# pca_result_50 = pipe.fit_transform(X_train[x_vars])

pca_result_50 = pca_50.fit_transform(X_train[x_vars])

print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50 .explained_variance_ratio_)))

# from sklearn.manifold import TSNE

# from matplotlib.ticker import NullFormatter

# from time import time

# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

# tsne_results = tsne.fit_transform(pca_result_50)



# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# ax.set_title("Perplexity=%d" % perplexity)

# ax.scatter(Y[red, 0], Y[red, 1], c="r")

# ax.scatter(Y[green, 0], Y[green, 1], c="g")

# ax.xaxis.set_major_formatter(NullFormatter())

# ax.yaxis.set_major_formatter(NullFormatter())

# ax.axis('tight')
# from sklearn.cluster import KMeans

# def process_data(train_df, test_df):

# #     logger.info('Features engineering - numeric data')

#     idx = [c for c in train_df.columns if c not in ['ID_code', 'target']]

#     for df in [test_df, train_df]:

#         for feat in idx:

#             df['r2_'+feat] = np.round(df[feat], 2)

#             df['r2_'+feat] = np.round(df[feat], 2)

#         df['sum'] = df[idx].sum(axis=1)  

#         df['min'] = df[idx].min(axis=1)

#         df['max'] = df[idx].max(axis=1)

#         df['mean'] = df[idx].mean(axis=1)

#         df['std'] = df[idx].std(axis=1)

#         df['skew'] = df[idx].skew(axis=1)

#         df['kurt'] = df[idx].kurtosis(axis=1)

#         df['med'] = df[idx].median(axis=1)

#         tmp=np.array(df[['var_0', 'var_2', 'var_26', 'var_76','var_81','var_139','var_191']])

#         kms=KMeans(n_clusters=30)

#         y=kms.fit_predict(tmp)

#         df['category'] = y

#         df['category'] = df['category'].astype('category')

#     print('Train and test shape:',train_df.shape, test_df.shape)

#     return train_df, test_df

# X_train1, X_test1 = process_data(raw_train, raw_test)

# X_train1.head(30)

# X_test1.head(30)
X_train1[['var_1','r2_var_1']].iloc[0]
# #Add PCA features:

# X_train = pd.concat([X_train, pd.DataFrame(train_pca,columns=['comp1_pca','comp2_pca'])],axis=1)

# #Add t-SNE features:

# # X_train = pd.concat([X_train, pd.DataFrame(tsne_results,columns=['comp1_tsne','comp2_tsne'])],axis=1)

# #get PCA test components:

# test_pca = pca.transform(X_test[x_vars])

# X_test = pd.concat([X_test, pd.DataFrame(test_pca,columns=['comp1_pca','comp2_pca'])],axis=1)

# # #get t-SNE test components:

# # test_pca_50 = pca_50.transform(X_test[x_vars])

# # test_tsne = tsne.transform(test_pca_50)

# # X_test = pd.concat([X_test, pd.DataFrame(test_pca,columns=['comp1_pca','comp2_pca'])],axis=1)

# # X_test = pd.concat([X_test, pd.DataFrame(test_tsne,columns=['comp1_tsne','comp2_tsne'])],axis=1)

# #check shape: (4 more columns added)

# X_train.shape, X_test.shape
# plt.figure(figsize=(80,80))

# sns.heatmap(X_train.corr(),

#            annot=True, fmt=".2f")
import warnings



from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

from imblearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectKBest

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression



# warnings.filterwarnings('ignore')

# param_grid = {'logisticregression__C': np.logspace(-1, 1, 7)}

# pipe = make_pipeline(RandomUnderSampler(), StandardScaler(), LogisticRegression(class_weight='balanced',

#                                                                                 random_state=0))

# model_01 = GridSearchCV(pipe, param_grid, cv=5)

# model_01.fit(X_train, y_train)



# score_01 = np.mean(cross_val_score(model_01, X_train, y_train, scoring='roc_auc', cv=7))

# print('Average ROC AUC Score: {:.3f}'.format(score_01))

# y_pred_01 = model_01.predict(X_test)

# #  print('   Test ROC AUC Score: {:.3f}'.format(roc_auc_score(y_test, y_pred)))

# y_test['target_01'] = y_pred_01

# y_test_01 = y_test[['ID_code','target_01']].copy()

# y_test_01.head()
# from xgboost import XGBClassifier



# model_02 = XGBClassifier(max_depth=2,

#                          learning_rate=1,

#                          min_child_weight = 1,

#                          subsample = 0.5,

#                          colsample_bytree = 0.1,

#                          scale_pos_weight = round(sum(y_train.target == 1)/len(y_train.target),2),

#                          #gamma=3,

#                          seed=0)

# model_02.fit(X_train, y_train.values)



# score_02 = np.mean(cross_val_score(model_02, X_train, y_train.values, scoring='roc_auc', cv=7))

# print('Average ROC AUC Score: {:.3f}'.format(score_02))
# y_pred_02 = model_02.predict(X_test)

# #  print('   Test ROC AUC Score: {:.3f}'.format(roc_auc_score(y_test, y_pred)))



# y_test['target'] = y_pred_02

# y_test_02 = y_test[['ID_code','target']].copy()

# y_test_02.head()

# y_test_02.to_csv('../input/sample_submission.csv', encoding='utf-8', index=False)
# from sklearn.ensemble import ExtraTreesClassifier



# model_08 = make_pipeline(RandomUnderSampler(), ExtraTreesClassifier(n_estimators=150,

#                                                                     criterion='entropy',

#                                                                     max_depth=8,

#                                                                     min_samples_split=300,

#                                                                     min_samples_leaf=15,

#                                                                     random_state=0,

#                                                                     class_weight='balanced_subsample'))

# model_08.fit(X_train, y_train)



# score_08 = np.mean(cross_val_score(model_08, X_train, y_train, scoring='roc_auc', cv=7))

# print('Average ROC AUC Score: {:.3f}'.format(score_08))