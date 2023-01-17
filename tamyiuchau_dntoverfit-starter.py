!ls ../input/300-probed-aucs-from-dont-overfit-ii

import pandas as pd

import numpy as np



import random

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression,RidgeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
prob_auc_df = pd.read_csv("../input/300-probed-aucs-from-dont-overfit-ii/probed_aucs.csv")

prob_auc_df.loc[:,"variable"]=prob_auc_df.loc[:,"variable"].transform(lambda x: x[1:]).astype(int)

prob_auc_df
prob_auc_df.sort_values("variable",inplace=True)

prob_auc_df.reset_index(drop=True,inplace=True)
prob_auc_df.loc[:,"public_auc"] = prob_auc_df.loc[:,"public_auc"]-0.5

prob_auc_df.loc[:,"public_auc"]=prob_auc_df.loc[:,"public_auc"].transform(lambda x: x if np.abs(x)>0.04 else 0)

prob_auc_df
score = np.array(prob_auc_df.loc[:,"public_auc"])
train_df = pd.read_csv("../input/dont-overfit-ii/train.csv").drop('id', axis=1)

train_df.head()
test_df = pd.read_csv('../input/dont-overfit-ii/test.csv').drop('id', axis = 1)

test_df.head()
from scipy.special import expit

prob_df = test_df.copy()

prob_df.loc[:,"target"]= expit(np.dot(np.array(prob_df),score))>0.5
prob_df.head()
#train_df,prob_df = prob_df,train_df
import matplotlib.pyplot as plt

import numpy as np

plt.figure(figsize=(12, 12))

import seaborn as sns #sns.set(style="whitegrid")

sns.violinplot(data=test_df.dropna()["0"],orient="h")
import scipy

#test_df = test_df.transpose()

for col in test_df.columns:

    z,p= scipy.stats.normaltest(test_df[col])

    print(p<0.005,z,p)
x_corr = test_df.corr().sort_values("0").sort_values("0",axis=1)#.sortlevel(level=0, inplace=True)#.iloc[0:20,0:20]

x_corr.head()
#x_corr.sort_values("0").sort_values("0",axis=1)
plt.figure(figsize=(20, 30))

mask = np.zeros_like(x_corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(10, 250, as_cmap=True)

sns.heatmap(x_corr,mask=mask,robust=True, vmin=-0.05,vmax=0.05, #cmap=cmap, 

        square=True,

        cbar_kws={"shrink": .5})

"""plt.bar(range(2), (train_df.shape[0], test_df.shape[0])) 

plt.xticks(range(2), ('Train', 'Test'))

plt.ylabel('Count') 

plt.show()"""
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

kbest = TSNE(3)

from sklearn.feature_selection import SelectKBest,RFE

from sklearn.feature_selection import f_regression

from sklearn.metrics import roc_auc_score

from sklearn.impute import SimpleImputer

from sklearn.svm import LinearSVC

from sklearn.preprocessing import QuantileTransformer,PowerTransformer,normalize,RobustScaler

from sklearn.model_selection import StratifiedShuffleSplit,LeavePOut,LeaveOneGroupOut

robust = RobustScaler().fit(np.concatenate((train_df.drop('target', axis=1), test_df), axis=0))

#kbest = PowerTransformer(method='yeo-johnson', standardize=True)#SelectKBest(f_regression,24)

#kbest.fit(test_df)

y = train_df['target']

X = SimpleImputer(strategy='mean').fit_transform(train_df.drop('target', axis=1))#,y)#robust.transform(

#kbest.transform(

#X = train_df.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,train_size=0.8)#

#X_train = X_test = X

#y_train = y_test = y
"""X_ = X_train

y_ = y_train

X = np.concatenate([X_train]*10)

y = np.concatenate([y_train]*10)"""
y_test
from sklearn.naive_bayes import GaussianNB

logreg = GaussianNB()#solver='liblinear',

logreg.fit(X_train, y_train)

test_score = logreg.score(X_test, y_test)

test_score
dir(logreg)
best_score = 0

best_std = 0

for penalty in ['l1']:

    for C in [2.**(i/2) for i in range(-9*2,6*2)]:#[0.001, 0.01, 0.1, 1, 10, 100]:

        score_ = []

        for train_index, val_index in StratifiedShuffleSplit(n_splits=12, test_size=0.1, random_state=42).split(X, y):

            X_train = X[train_index]

            y_train = y[train_index]

            X_test = X[val_index]

            y_test = y[val_index]

            seed = random.randint(0,2<<31)

            np.random.seed(seed)

            logreg = LogisticRegression(dual=False,max_iter=10**5,penalty=penalty, C=C)#, solver='liblinear')

            logreg.fit(X_train, y_train)

            score_ += [roc_auc_score(y_train,logreg.predict(X_train))]

        score = np.mean(score_)

        if score > best_score:

                best_std = np.std(score_)

                print(best_score,best_std)

                best_score = score

                best_parameters = {'C': C, 'penalty': penalty}

                s = seed
best_parameters
#best_parameters = {'C': 0.1, 'penalty': 'l1'}
reg_list = []

for train_index, val_index in StratifiedShuffleSplit(n_splits=30, test_size=0.1, random_state=42).split(X, y):

    X_train = X[train_index]

    y_train = y[train_index]

    X_test = X[val_index]

    y_test = y[val_index]

    seed = random.randint(0,2<<31)

    np.random.seed(seed)

    logreg = LogisticRegression(dual=False,max_iter=10**5,**best_parameters)#solver='liblinear',

    logreg.fit(X_train, y_train)

    test_score = logreg.score(X_test, y_test)

    reg_list.append(logreg)

#flattern)weight

def auc_reg():

    return sum(map(lambda i:i.score(X_test, y_test), reg_list))/len(reg_list)

def score_reg():

    return sum(map(lambda i:roc_auc_score(y_test,i.predict(X_test)), reg_list))/len(reg_list)

def sum_reg(x):

    return sum(map(lambda i:i.predict_proba(x), reg_list))/len(reg_list)

test_score = score_reg()
X_prob = prob_df.drop("target",axis=1)

Y_prob = prob_df["target"]
sum(map(lambda i:roc_auc_score(Y_prob,i.predict(X_prob)), reg_list))/len(reg_list)
x = np.average([i.coef_ for i in reg_list],axis = 0)

print(x,np.std(x),np.median(np.std([i.coef_ for i in reg_list],axis = 0)))

logreg.coef_ = x

logreg.intercept_ = np.average([i.intercept_ for i in reg_list],axis = 0)

print(logreg.score(X_test,y_test))

print(roc_auc_score(y_test,logreg.predict(X_test)))

print(roc_auc_score(y_train,logreg.predict(X_train)))
print(roc_auc_score(Y_prob,logreg.predict(X_prob)))
print("Best score: {:.3f} {:.3f}".format(best_score,best_std))

print("Best parameters: {}".format(best_parameters))

print("Best score on test data: {:.3f}".format(test_score))
sub = pd.read_csv('../input/dont-overfit-ii/sample_submission.csv')

sub['target'] = sum_reg(robust.transform(test_df))[:,1]#logreg.predict_proba(test_df)[:,1]

#sub['target'] = sum_reg(test_df)[:,1]#logreg.predict_proba(test_df)[:,1]

#sub['target'] = logreg.predict_proba(robust.transform(test_df))#[:,1]#kbest.transform(

#sub['target'] = logreg.predict(test_df)#[:,1]

sub.to_csv('submission.csv', index=False)
sub.head(100)
#logreg.score(test_df)

#print(test_df)