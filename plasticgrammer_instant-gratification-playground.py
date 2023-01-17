import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import gc,os,sys



from sklearn import metrics, preprocessing

from sklearn.decomposition import PCA, KernelPCA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.feature_selection import RFE, RFECV, VarianceThreshold

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import NuSVC

from tqdm import tqdm



sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format



print(os.listdir("../input"))
%%time

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape, test.shape)
train.head()
null_cnt = train.isnull().sum().sort_values()

print('null count:', null_cnt[null_cnt > 0])
c = train['target'].value_counts().to_frame()

c.plot.bar()

print(c)
fig, ax = plt.subplots(1, 3, figsize=(16,3), sharey=True)



train['muggy-smalt-axolotl-pembus'].hist(bins=50, ax=ax[0])

train['dorky-peach-sheepdog-ordinal'].hist(bins=50, ax=ax[1])

train['slimy-seashell-cassowary-goose'].hist(bins=50, ax=ax[2])
for col in train.columns:

    unicos = train[col].unique().shape[0]

    if unicos < 1000:

        print(col, unicos)
train['wheezy-copper-turtle-magic'].hist(bins=128, figsize=(12,3))

#test['wheezy-copper-turtle-magic'].hist(bins=128, figsize=(12,3))
print(train['wheezy-copper-turtle-magic'].describe())

print()

print('unique value count:', train['wheezy-copper-turtle-magic'].nunique())
numcols = train.drop(['id','target','wheezy-copper-turtle-magic'],axis=1).select_dtypes(include='number').columns.values
pca = PCA()

pca.fit(train[numcols])

plt.xlabel('components')

plt.plot(np.add.accumulate(pca.explained_variance_ratio_))

plt.show()
X_subset = train[train['wheezy-copper-turtle-magic'] == 0][numcols]

pca.fit(X_subset)

plt.xlabel('components')

plt.plot(np.add.accumulate(pca.explained_variance_ratio_))

plt.show()
from sklearn.neighbors import KNeighborsClassifier



X_subset = train[train['wheezy-copper-turtle-magic'] == 1][numcols]

Y_subset = train[train['wheezy-copper-turtle-magic'] == 1]['target']



for k in range(2, 10):

    knc = KNeighborsClassifier(n_neighbors=k)

    knc.fit(X_subset, Y_subset)

    score = knc.score(X_subset, Y_subset)

    print("[{}] score: {:.2f}".format(k, score))
filters = [('StandardScaler', StandardScaler()),

           ('MinMaxScaler', MinMaxScaler()),

           ('PCA', PCA(n_components=0.98)),

           ('KernelPCA(poly)', KernelPCA(kernel="poly", degree=3, gamma=15)),

           ('KernelPCA(rbf)', KernelPCA(kernel="rbf")),

           ]

for name, f in filters:

    X_subset2 = f.fit_transform(X_subset)

    knc = KNeighborsClassifier(n_neighbors=3)

    knc.fit(X_subset2, Y_subset)

    score = knc.score(X_subset2, Y_subset)

    print("[{}] score: {:.2f}".format(name, score))
from sklearn.ensemble import BaggingClassifier



knn3 = KNeighborsClassifier(n_neighbors=3)

knnb = BaggingClassifier(base_estimator=knn3, n_estimators=20)

knnb.fit(X_subset, Y_subset)

score = knnb.score(X_subset, Y_subset)

print("score: {:.2f}".format(score))
all_data = train.append(test, sort=False).reset_index(drop=True)

del train, test

gc.collect()



all_data.head()
# drop constant column

constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]

print('drop columns:', constant_column)

all_data.drop(constant_column, axis=1, inplace=True)
corr_matrix = all_data.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]

del upper



drop_column = all_data.columns[to_drop]

print('drop columns:', drop_column)

#all_data.drop(drop_column, axis=1, inplace=True)
X_train = all_data[all_data['target'].notnull()].reset_index(drop=True)

X_test = all_data[all_data['target'].isnull()].drop(['target'], axis=1).reset_index(drop=True)

del all_data

gc.collect()



# drop ID_code

X_train.drop(['id'], axis=1, inplace=True)

X_test_ID = X_test.pop('id')



Y_train = X_train.pop('target')



print(X_train.shape, X_test.shape)
oof_preds = np.zeros(X_train.shape[0])

sub_preds = np.zeros(X_test.shape[0])



splits = 11



for i in tqdm(range(512)):

    train2 = X_train[X_train['wheezy-copper-turtle-magic'] == i][numcols]

    test2 = X_test[X_test['wheezy-copper-turtle-magic'] == i][numcols]

    train2_y = Y_train[train2.index]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    sel = VarianceThreshold(threshold=1.5)

    train2 = sel.fit_transform(train2)

    test2 = sel.transform(test2)    

    

    skf = StratifiedKFold(n_splits=splits, random_state=42)

    for train_index, test_index in skf.split(train2, train2_y):

        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        clf.fit(train2[train_index], train2_y.iloc[train_index])

        oof_preds[idx1[test_index]] = clf.predict_proba(train2[test_index])[:,1]

        sub_preds[idx2] += clf.predict_proba(test2)[:,1] / skf.n_splits
error = X_train[((Y_train == 1) & (oof_preds < 0.5)) | ((Y_train == 0) & (oof_preds >= 0.5))]

print('error rate: %.3f%%' % (len(error) / X_train.shape[0] * 100))
fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)

auc = metrics.auc(fpr, tpr)



plt.plot(fpr, tpr, label='ROC curve (area = %.3f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
X_test_p1 = X_test[(sub_preds <= 0.01)].copy()

X_test_p2 = X_test[(sub_preds >= 0.99)].copy()

X_test_p1['target'] = 0

X_test_p2['target'] = 1

print(X_test_p1.shape, X_test_p2.shape)



Y_train = pd.concat([Y_train, X_test_p1.pop('target'), X_test_p2.pop('target')], axis=0)

X_train = pd.concat([X_train, X_test_p1, X_test_p2], axis=0)

Y_train.reset_index(drop=True, inplace=True)

X_train.reset_index(drop=True, inplace=True)
_='''

for i in range(512):

    train_f = (X_train['wheezy-copper-turtle-magic'] == i)

    test_f = (X_test['wheezy-copper-turtle-magic'] == i)

    X_train_sub = X_train[train_f][numcols]

    Y_train_sub = Y_train[train_f]

    X_test_sub = X_test[test_f][numcols]



    lda = LinearDiscriminantAnalysis(n_components=1)

    lda.fit(X_train_sub, Y_train_sub)

    X_train.loc[train_f, 'lda'] = lda.transform(X_train_sub).reshape(-1)

    X_test.loc[test_f, 'lda'] = lda.transform(X_test_sub).reshape(-1)

    

    knc = KNeighborsClassifier(n_neighbors=3)

    knc.fit(X_train_sub, Y_train_sub)

    X_train.loc[train_f, 'knc'] = knc.predict_proba(X_train_sub)[:,1]

    X_test.loc[test_f, 'knc'] = knc.predict_proba(X_test_sub)[:,1]

'''
splits = 11



Y_train_org = Y_train.copy()

X_train_org = X_train.copy()



for itr in range(4):

    X_test_p1 = X_test[(sub_preds <= 0.04)].copy()

    X_test_p2 = X_test[(sub_preds >= 0.96)].copy()

    X_test_p1['target'] = 0

    X_test_p2['target'] = 1

    print(X_test_p1.shape, X_test_p2.shape)



    Y_train = pd.concat([Y_train_org, X_test_p1.pop('target'), X_test_p2.pop('target')], axis=0)

    X_train = pd.concat([X_train_org, X_test_p1, X_test_p2], axis=0)

    Y_train.reset_index(drop=True, inplace=True)

    X_train.reset_index(drop=True, inplace=True)



    oof_preds = np.zeros(X_train.shape[0])

    sub_preds = np.zeros(X_test.shape[0])

        

    for i in tqdm(range(512)):

        train2 = X_train[X_train['wheezy-copper-turtle-magic'] == i][numcols]

        train2_y = Y_train[train2.index]

        test2 = X_test[X_test['wheezy-copper-turtle-magic'] == i][numcols]

        idx1 = train2.index; idx2 = test2.index

        train2.reset_index(drop=True,inplace=True)



        sel = VarianceThreshold(threshold=1.5)

        train2 = pd.DataFrame(sel.fit_transform(train2))

        test2 = pd.DataFrame(sel.transform(test2))

        

        sel = StandardScaler()

        train2 = pd.DataFrame(sel.fit_transform(train2))

        test2 = pd.DataFrame(sel.transform(test2))



        skf = StratifiedKFold(n_splits=splits, random_state=42)

        for train_index, test_index in skf.split(train2, train2_y):

            #clf = KNeighborsClassifier(n_neighbors=3)

            clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=42, nu=0.59, coef0=0.053)

            #clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

            #clf = BaggingClassifier(base_estimator=clf, n_estimators=30)

            clf.fit(train2.iloc[train_index], train2_y.iloc[train_index])

            oof_preds[idx1[test_index]] += clf.predict_proba(train2.iloc[test_index])[:,1]

            sub_preds[idx2] += clf.predict_proba(test2)[:,1] / skf.n_splits
#oof_preds[oof_preds <= 0.01] = 0

#oof_preds[oof_preds >= 0.99] = 1
fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)

auc = metrics.auc(fpr, tpr)



plt.plot(fpr, tpr, label='ROC curve (area = %.3f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
#sub_preds[sub_preds <= 0.01] = 0

#sub_preds[sub_preds >= 0.99] = 1
submission = pd.DataFrame({

    'id': X_test_ID,

    'target': sub_preds

})

submission.to_csv("submission.csv", index=False)
submission['target'].hist(bins=25, alpha=0.6)

print(submission['target'].sum() / len(submission))
submission.head()