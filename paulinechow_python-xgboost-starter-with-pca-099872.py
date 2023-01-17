import numpy as np

import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import matthews_corrcoef, roc_auc_score

from sklearn.cross_validation import cross_val_score, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.cross_validation import StratifiedKFold



from sklearn import preprocessing 

from sklearn.decomposition import PCA 

from sklearn.metrics import classification_report



import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

matplotlib.style.use('ggplot') 
# I'm limited by RAM here and taking the first N rows is likely to be

# a bad idea for the date data since it is ordered.

# Sample the data in a roundabout way:

date_chunks = pd.read_csv("../input/train_date.csv", index_col=0, chunksize=100000, dtype=np.float32)

num_chunks = pd.read_csv("../input/train_numeric.csv", index_col=0,

                         usecols=list(range(969)), chunksize=100000, dtype=np.float32)

X = pd.concat([pd.concat([dchunk, nchunk], axis=1).sample(frac=0.05)

               for dchunk, nchunk in zip(date_chunks, num_chunks)])

y = pd.read_csv("../input/train_numeric.csv", index_col=0, usecols=[0,969], dtype=np.float32).fillna(0).loc[X.index].values.ravel()
X = X.fillna(0)

X = X.values
X3 = X #set varible for code below
# Only need to run once

# n components will explain up to 70% of the data

#PCA_A = PCA(n_components=X3.shape[1]-1)

#PCA_Y = PCA_A.fit_transform(X3)

#

#print(PCA_A.n_components)

#print(PCA_A.explained_variance_[:15])

#x_vals = np.arange(1, PCA_A.n_components)
#tmp =0 

#for i, p in enumerate(PCA_A.explained_variance_ratio_):

#    tmp = sum(PCA_A.explained_variance_ratio_[:i])

#    if tmp >=.70 :

#        print('up until component #:', i)

#       #print np.cumsum(PCA_A.explained_variance_ratio_[:i])

#        break
#plt.figure(figsize=(7,5))



#plt.plot(range(X3.shape[1]-1), PCA_A.explained_variance_ratio_, '-o', label='Individual component')

#plt.plot(range(X3.shape[1]-1), np.cumsum(PCA_A.explained_variance_ratio_), '-s', label='Cumulative')



#plt.ylabel('Proportion of Variance Explained')

#plt.xlabel('Principal Component')

#plt.xlim(0.75, 20)

#plt.ylim(0,1.05)

#plt.xticks(range(20))

#plt.legend(loc=2);
#Create a pipeline for PCA & XGBClassifier

pipe_boosch = Pipeline(steps=[('pca', PCA()), 

                            ('xgb', XGBClassifier()) ])



cvs = StratifiedKFold(y, n_folds = 3, shuffle=True)

pipe_boosch.set_params(pca__n_components=100, xgb__base_score=0.005)

print(cross_val_score(pipe_boosch, X3, y, cv=cvs, n_jobs=-1))
clf = XGBClassifier(base_score=0.005)

clf.fit(X3, y)
# threshold for a manageable number of features

plt.hist(clf.feature_importances_[clf.feature_importances_>0])

important_indices = np.where(clf.feature_importances_>0.001)[0]

print(important_indices)
# load entire dataset for these features. 

# note where the feature indices are split so we can load the correct ones straight from read_csv

n_date_features = 1156

X_new = np.concatenate([

    pd.read_csv("../input/train_date.csv", index_col=0, dtype=np.float32,

                usecols=np.concatenate([[0], important_indices[important_indices < n_date_features] + 1])).fillna(0).values,

    pd.read_csv("../input/train_numeric.csv", index_col=0, dtype=np.float32,

                usecols=np.concatenate([[0], important_indices[important_indices >= n_date_features] + 1 - 1156])).fillna(0).values

], axis=1)

y_new = pd.read_csv("../input/train_numeric.csv", index_col=0, dtype=np.float32, usecols=[0,969]).values.ravel()
clf_pipeline = Pipeline(steps=[('pca', PCA()), 

                            ('xgb', XGBClassifier()) ])

cvs = StratifiedKFold(y_new, n_folds = 2)

preds = np.ones(y_new.shape[0])

clf_pipeline.set_params(pca__n_components=20, xgb__base_score=0.005).fit(X_new, y_new)



#for i, (train, test) in enumerate(cvs):

#    preds[test] = clf_pipeline.fit(X_new[train], y_new[train]).predict_proba(X[test])[:,1]

#    print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[test], preds[test])))

#print(roc_auc_score(y_new, preds))
# pick the best threshold out-of-fold

thresholds = np.linspace(0.01, 0.99, 50)

mcc = np.array([matthews_corrcoef(y_new, preds>thr) for thr in thresholds])

plt.plot(thresholds, mcc)

best_threshold = thresholds[mcc.argmax()]

print(mcc.max())
# load test data

X_test = np.concatenate([

    pd.read_csv("../input/test_date.csv", index_col=0, dtype=np.float32,

                usecols=np.concatenate([[0], important_indices[important_indices<1156]+1])).fillna(0).values,

    pd.read_csv("../input/test_numeric.csv", index_col=0, dtype=np.float32,

                usecols=np.concatenate([[0], important_indices[important_indices>=1156] +1 - 1156])).fillna(0).values

], axis=1)
# generate predictions at the chosen threshold

preds = (clf_pipeline.predict_proba(X_test)[:,1] > best_threshold).astype(np.int8)
# and submit

sub = pd.read_csv("../input/sample_submission.csv", index_col=0)

sub["Response"] = preds

sub.to_csv("submission.csv.gz", compression="gzip")