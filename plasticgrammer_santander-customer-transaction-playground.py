import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import gc,os,sys



sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format



print(os.listdir("../input"))
%%time

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape, test.shape)
for c in train.columns:

    if c not in test.columns: print(c)
train.head()
null_cnt = train.isnull().sum().sort_values()

print('null count:', null_cnt[null_cnt > 0])
numcols = train.drop('target',axis=1).select_dtypes(include='number').columns.values
train['target'].value_counts().to_frame().plot.bar()
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
cols = [col for col in all_data.columns if col not in ['ID_code']]

for i, t in all_data.loc[:, cols].dtypes.iteritems():

    if t == object:

        print(i)

        all_data[i] = pd.factorize(all_data[i])[0]
from sklearn import preprocessing



#scaler = preprocessing.StandardScaler()

#scaler = preprocessing.MaxAbsScaler()

scaler = preprocessing.RobustScaler()

all_data.loc[:,numcols] = scaler.fit_transform(all_data[numcols])
_='''noneffective

feats = ["var_{}".format(i) for i in range(200)]

for f in feats:

    all_data[f] = pd.cut(all_data[f], 100, labels=range(100))

'''
all_data.head()
mean1 = all_data[all_data['target']==1].mean()

mean0 = all_data[all_data['target']==0].mean()

pd.concat([mean1, mean0, np.abs(mean1-mean0)], axis=1).sort_values(by=2, ascending=False)[1:10]
X_train = all_data[all_data['target'].notnull()].reset_index(drop=True)

X_test = all_data[all_data['target'].isnull()].drop(['target'], axis=1).reset_index(drop=True)

del all_data

gc.collect()



# drop ID_code

X_train.drop(['ID_code'], axis=1, inplace=True)

X_test_ID = X_test.pop('ID_code')



Y_train = X_train.pop('target')



print(X_train.shape, X_test.shape)
_='''

def get_redundant_pairs(df):

    # Get diagonal and lower triangular pairs of correlation matrix

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations")

print(get_top_abs_correlations(X_train))

'''
_='''noneffective

from imblearn.over_sampling import SMOTE,ADASYN

#from imblearn.combine import SMOTETomek



sm = SMOTE(random_state=42)

#sm = SMOTE(kind='svm',random_state=42)

#sm = SMOTE(kind='borderline1',random_state=42)

#sm = ADASYN(random_state=42)

#sm = SMOTETomek(random_state=42)

X_train, Y_train = sm.fit_sample(X_train, Y_train)

X_train = pd.DataFrame(X_train, columns=X_test.columns)

print(X_train.shape)

'''
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, init='k-means++', max_iter=3000, random_state=42)

X_train_km = km.fit_predict(X_train)

X_test_km = km.predict(X_test)



km_data = pd.DataFrame({'KMeans':X_train_km, 'target':Y_train})

sns.countplot(x='KMeans', hue='target', palette='Set1', data=km_data)
from sklearn.decomposition import PCA, KernelPCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold

from sklearn import metrics



pca = PCA()

pca.fit(X_train)

ev_ratio = pca.explained_variance_ratio_

ev_ratio = np.hstack([0,ev_ratio.cumsum()])



plt.xlabel('components')

plt.plot(ev_ratio)

plt.show()
_='''

def select_n_components(var_ratio, goal_var: float) -> int:

    total_variance = 0.0

    n_components = 0

    for explained_variance in var_ratio:

        total_variance += explained_variance

        n_components += 1

        if total_variance >= goal_var:

            break

            

    return n_components



lda = LDA(n_components=None)

lda.fit(X_train, Y_train)

print(select_n_components(lda.explained_variance_ratio_, 0.95))

'''
fig, ax = plt.subplots(2, 2, figsize=(16, 6))

ax = ax.ravel()



pca = PCA(n_components=1)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.fit_transform(X_test)

pca_1 = X_train_pca[Y_train > 0].reshape(-1)

pca_0 = X_train_pca[Y_train == 0].reshape(-1)

ax[0].hist([pca_1, pca_0], color=['b','r'], bins=30, alpha=0.5, histtype='barstacked')

ax[0].set_title('PCA visualization')



lda = LDA(n_components=1)

lda.fit(X_train, Y_train)

X_train_lda = lda.transform(X_train)

X_test_lda = lda.transform(X_test)

lda_1 = X_train_lda[Y_train > 0].reshape(-1)

lda_0 = X_train_lda[Y_train == 0].reshape(-1)

ax[1].hist([lda_1, lda_0], color=['b','r'], bins=30, alpha=0.5, histtype='barstacked')

ax[1].set_title('LDA visualization')



ax[2].hist(X_test_pca, color='g', bins=30, alpha=0.5, histtype='barstacked')

ax[3].hist(X_test_lda, color='g', bins=30, alpha=0.5, histtype='barstacked')



plt.show()
fig, ax = plt.subplots(2, 2, figsize=(16, 6))

ax = ax.ravel()



gnb = GaussianNB()

gnb.fit(X_train, Y_train)

X_train_gnb = gnb.predict_log_proba(X_train)[:,1]

X_test_gnb = gnb.predict_log_proba(X_test)[:,1]

gnb_1 = X_train_gnb[Y_train > 0].reshape(-1)

gnb_0 = X_train_gnb[Y_train == 0].reshape(-1)

ax[0].hist([gnb_1, gnb_0], color=['b','r'], bins=30, alpha=0.5, histtype='barstacked')

ax[0].set_title('GaussianNB visualization')



bnb = BernoulliNB(fit_prior=True)

bnb.fit(X_train, Y_train)

X_train_bnb = bnb.predict_log_proba(X_train)[:,1]

X_test_bnb = bnb.predict_log_proba(X_test)[:,1]

bnb_1 = X_train_bnb[Y_train > 0].reshape(-1)

bnb_0 = X_train_bnb[Y_train == 0].reshape(-1)

ax[1].hist([bnb_1, bnb_0], color=['b','r'], bins=30, alpha=0.5, histtype='barstacked')

ax[1].set_title('BernoulliNB visualization')



ax[2].hist(X_test_gnb, color='g', bins=30, alpha=0.5, histtype='barstacked')

ax[3].hist(X_test_bnb, color='g', bins=30, alpha=0.5, histtype='barstacked')



plt.show()
fig, ax = plt.subplots(2, 2, figsize=(16, 6))

ax = ax.ravel()



lgr = LogisticRegression(C=0.1, class_weight='balanced', penalty='l1', solver='liblinear')

lgr.fit(X_train, Y_train)

X_train_lgr = lgr.predict_log_proba(X_train)[:,1]

X_test_lgr = lgr.predict_log_proba(X_test)[:,1]

lgr_1 = X_train_lgr[Y_train > 0].reshape(-1)

lgr_0 = X_train_lgr[Y_train == 0].reshape(-1)

ax[0].hist([lgr_1, lgr_0], color=['b','r'], bins=30, alpha=0.5, histtype='barstacked')

ax[0].set_title('LogisticRegression visualization')

ax[2].hist(X_test_lgr, color='g', bins=30, alpha=0.5, histtype='barstacked')



sgd = SGDClassifier(max_iter=10000, loss='log', tol=1e-5)

sgd.fit(X_train, Y_train)

X_train_sgd = sgd.predict_log_proba(X_train)[:,1]

X_test_sgd = sgd.predict_log_proba(X_test)[:,1]

sgd_1 = X_train_sgd[Y_train > 0].reshape(-1)

sgd_0 = X_train_sgd[Y_train == 0].reshape(-1)

ax[1].hist([sgd_1, sgd_0], color=['b','r'], bins=30, alpha=0.5, histtype='barstacked')

ax[1].set_title('SGDClassifier visualization')

ax[3].hist(X_test_sgd, color='g', bins=30, alpha=0.5, histtype='barstacked')

plt.show()
fpr, tpr, thresholds = metrics.roc_curve(Y_train, X_train_gnb)

auc = metrics.auc(fpr, tpr)



plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



scores = cross_val_score(pca, X_train, Y_train, cv=skf)

print("PCA, Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    

scores = cross_val_score(lda, X_train, Y_train, cv=skf) # max components is (classes - 1)

print("LDA, Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    

scores = cross_val_score(gnb, X_train, Y_train, cv=skf)

print("GNB, Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))



scores = cross_val_score(bnb, X_train, Y_train, cv=skf)

print("BNB, Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))



scores = cross_val_score(lgr, X_train, Y_train, cv=skf)

print("LGR, Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))



scores = cross_val_score(sgd, X_train, Y_train, cv=skf)

print("SGD, Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
_='''noneffective

from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)

X_all_tsne = tsne.fit_transform(pd.concat([n_train,n_test]))

X_train_tsne = X_all_tsne[:n_train.shape[0]]

X_test_tsne = X_all_tsne[n_train.shape[0]:]

print(X_train_tsne.shape)



plt.figure(figsize=(6,6))

plt.scatter(X_train_tsne[:,0], X_train_tsne[:,1], c=color, alpha=0.3)

plt.title("t-SNE visualization")

plt.show()



n_train['tnse1'] = X_train_tsne[:,0]

n_test['tnse1'] = X_test_tsne[:,0]

n_train['tnse2'] = X_train_tsne[:,1]

n_test['tnse2'] = X_test_tsne[:,1]

'''
for df in [X_train, X_test]:

    df['sum'] = df.sum(axis=1)  

    df['min'] = df.min(axis=1)

    df['max'] = df.max(axis=1)

    df['mean'] = df.mean(axis=1)

    df['std'] = df.std(axis=1)

    df['skew'] = df.skew(axis=1)

    df['kurt'] = df.kurtosis(axis=1)

    df['med'] = df.median(axis=1)

    df['var'] = df.var(axis=1)

    df['negval'] = df.apply(lambda x: (x < 0).astype(int).sum(), axis=1)
n_train = pd.DataFrame()

n_test = pd.DataFrame()



#n_train['_km'] = X_train_km

#n_test['_km'] = X_test_km

#n_train['_lda'] = X_train_lda

#n_test['_lda'] = X_test_lda

#n_train['_pca'] = X_train_pca

#n_test['_pca'] = X_test_pca

n_train['_lgr'] = X_train_lgr

n_test['_lgr'] = X_test_lgr

n_train['_sgd'] = X_train_sgd

n_test['_sgd'] = X_test_sgd

n_train['_gnb'] = X_train_gnb

n_test['_gnb'] = X_test_gnb

#n_train['_bnb'] = X_train_bnb

#n_test['_bnb'] = X_test_bnb
_='''

for c in X_train.columns:

    X_train[c +'_perc'] = X_train[c].rank()

    X_test[c + '_perc'] = X_test[c].rank()



from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(2)

n_train = poly.fit_transform(n_train)

n_test = poly.transform(n_test)



for c in n_train.columns:

    n_train[c+'_r'] = rankdata(n_train[c]).astype('float32')

    n_train[c+'_n'] = norm.cdf(n_train[c]).astype('float32')

    n_test[c+'_r'] = rankdata(n_test[c]).astype('float32')

    n_test[c+'_n'] = norm.cdf(n_test[c]).astype('float32')

'''
X_train = pd.concat([X_train, n_train], axis=1)

X_test = pd.concat([X_test, n_test], axis=1)
_='''noneffective

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif



feat = SelectKBest(f_classif, k=50)

feat.fit(X_train, Y_train)



X_train = X_train.loc[:,feat.get_support()]

X_test = X_test.loc[:,feat.get_support()]

'''
print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:5])
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb



params = {

    "objective" : "binary",

    "metric" : "auc",

    "boosting": 'gbdt',

    "max_depth" : -1,

    "num_leaves" : 13,

    "learning_rate" : 0.01,

    "bagging_freq": 5,

    "bagging_fraction" : 0.4,

    "feature_fraction" : 0.05,

    "min_data_in_leaf": 80,

    "min_sum_heassian_in_leaf": 10,

    "tree_learner": "serial",

    "boost_from_average": "false",

    "bagging_seed" : 42,

    "verbosity" : 1,

    "seed": 42

}



folds = StratifiedKFold(n_splits=10)

oof_preds = np.zeros(X_train.shape[0])

sub_preds = np.zeros(X_test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, Y_train)):

    trn_x, trn_y = X_train.iloc[trn_], Y_train[trn_]

    val_x, val_y = X_train.iloc[val_], Y_train[val_]



    model = lgb.LGBMRegressor(**params, n_estimators=100000)

    model.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=3000, verbose=1000)



    oof_preds[val_] = model.predict(val_x, num_iteration=model.best_iteration_)

    sub_preds += model.predict(X_test, num_iteration=model.best_iteration_) / folds.n_splits
# Plot feature importance

feature_importance = model.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = sorted_idx[len(feature_importance) - 20:]

pos = np.arange(sorted_idx.shape[0]) + .5



plt.figure(figsize=(12,6))

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)

auc = metrics.auc(fpr, tpr)



plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.grid(True)
_='''

from catboost import CatBoostClassifier



folds = StratifiedKFold(n_splits=5)

oof2_preds = np.zeros(X_train.shape[0])

sub2_preds = np.zeros(X_test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, Y_train)):

    trn_x, trn_y = X_train.iloc[trn_], Y_train[trn_]

    val_x, val_y = X_train.iloc[val_], Y_train[val_]

    

    model = CatBoostClassifier(iterations=100000, learning_rate=0.01, objective="Logloss", eval_metric='AUC')

    model.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=3000, verbose=1000)



    oof2_preds[val_] = model.predict_proba(val_x)[:,1]

    sub2_preds += model.predict_proba(X_test)[:,1] / folds.n_splits

'''
#preds = sub_preds * 0.7 + sub2_preds * 0.3

preds = sub_preds
submission = pd.DataFrame({

    'ID_code': X_test_ID,

    'target': preds

})

submission.to_csv("submission.csv", index=False)
submission['target'].sum()