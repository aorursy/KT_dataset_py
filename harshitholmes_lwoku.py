import numpy as np

import pandas as pd

import pandas_profiling as pp

import matplotlib.pyplot as plt



import os 

import math 

import warnings as wn

import seaborn as ss



from sklearn.metrics import *

from sklearn.model_selection import *

from sklearn.pipeline import *

from sklearn.feature_selection import *

from sklearn.ensemble import *

from sklearn.preprocessing import *



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.linear_model import LogisticRegression as LR

from sklearn.naive_bayes import GaussianNB as GNB

from sklearn.svm import SVC 

from sklearn.neighbors import KNeighborsClassifier as KNC



from sklearn.multiclass import OneVsRestClassifier

from xgboost import XGBClassifier

from mlxtend.classifier import StackingCVClassifier



wn.filterwarnings('ignore')
trn = pd.read_csv('../input/learn-together/train.csv')

tst = pd.read_csv('../input/learn-together/test.csv')



print(trn.shape)

print(tst.shape)
trn.head()
tst.head()
xtrn = trn.drop(['Cover_Type'], axis = 1)

ytrn = trn['Cover_Type']



train_X, test_X, train_y, test_y = train_test_split(xtrn, ytrn, test_size = 0.25, random_state = 33)



xtrn = trn.drop(['Id'], axis = 1)

train_X = train_X.drop(['Id'], axis = 1)



print(train_X, test_X)

print(train_y, test_y)
ss.distplot(train_X['Elevation'], label = 'train_X')

ss.distplot(test_X['Elevation'], label = 'test_X')

ss.distplot(tst['Elevation'], label = 'tst')

plt.legend()

plt.title('Elevation')

plt.show()
ss.distplot(train_X['Aspect'], label = 'train_X')

ss.distplot(test_X['Aspect'], label = 'test_X')

ss.distplot(tst['Aspect'], label = 'tst')

plt.title('Aspect')

plt.legend()

plt.show()
tgt = trn.Cover_Type.value_counts()

ss.countplot(x = 'Cover_Type', data = trn)

plt.title('Class Distribution');

print(tgt)
N = xtrn.shape[1]

nrows = int(N/math.sqrt(N))

ncols = math.ceil(N/nrows)



fig, axs = plt.subplots(nrows, ncols, figsize = (20,24))

for i, column in enumerate(xtrn.columns):

    r = int(i/ncols)

    c = i % ncols

    axs[r, c].scatter(xtrn.iloc[:, i], ytrn, alpha=0.2, s=9)

    axs[r, c].set_title(column[:min(20, len(column))])

    

plt.show()
classifier_rf = RandomForestClassifier(n_estimators = 625,

                                       max_features = 0.4,

                                       max_depth = 575,

                                       min_samples_split = 2,

                                       min_samples_leaf = 1,

                                       bootstrap = False,

                                       random_state = 42)

classifier_xgb = OneVsRestClassifier(XGBClassifier(random_state = 33))

classifier_et = ExtraTreesClassifier(random_state = 42)
sc = StackingCVClassifier(classifiers=[classifier_rf,

                                         classifier_xgb,

                                         classifier_et],

                            use_probas = True,

                            meta_classifier = classifier_rf)







labels = ['Random Forest', 'XGBoost', 'ExtraTrees', 'MetaClassifier']









for cf, label in zip([classifier_rf, classifier_xgb, classifier_et, classifier_rf], labels):

    scores = cross_val_score(cf, train_X.values, train_y.values.ravel(),

                             cv = 5,

                             scoring = 'accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
sc.fit(train_X.values, train_y.values.ravel())
vid = trn['Id']

vpd = sc.predict(train_X.values)
acc = accuracy_score(train_y, vpd)

print(acc)
scff = StackingCVClassifier(classifiers=[classifier_rf,

                                            classifier_xgb,

                                            classifier_et],

                               use_probas=True,

                               meta_classifier=classifier_rf)



scff.fit(xtrn.values, ytrn.values.ravel())
tstid = tst['Id']

tstpd = scff.predict(tst.values)
rs = pd.DataFrame({'Id': tstid,

                   'Cover_Type': tstpd})

rs.to_csv('submission.csv', index = False)