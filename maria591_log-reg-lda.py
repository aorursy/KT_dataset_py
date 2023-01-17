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
import pandas as pd

import numpy as np

from scipy import interp

from sklearn.metrics import confusion_matrix

import seaborn as sns

import math

import gc

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn import preprocessing

import imblearn

from mpl_toolkits.mplot3d import Axes3D

np.random.seed(5)

from sklearn import decomposition, datasets 

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.metrics import auc

from sklearn import metrics



from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.utils import shuffle



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.preprocessing import MinMaxScaler



from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
train_df = pd.read_csv('../input/preprocessed-home-credit/local_data_final.csv')

test_df = pd.read_csv('../input/preprocessed-home-credit/kaggle_test_final.csv')
result = pd.DataFrame()

result['SK_ID_CURR'] = test_df['SK_ID_CURR'].values.tolist()

X_train1 = (train_df.drop('TARGET', axis = 1)).head(100000)

Y_train = train_df['TARGET'].head(100000)

X_test1 = (test_df.drop('TARGET', axis = 1))
del train_df

del test_df
from sklearn.preprocessing import Imputer

imp = Imputer(strategy='mean').fit(X_train1)

X_train = pd.DataFrame((imp.transform(X_train1)).tolist())

X_test = pd.DataFrame((imp.transform(X_test1)).tolist())
X_train.columns = X_train1.columns

X_test.columns = X_train1.columns



X_train.shape

X_train.head()
del X_train1

del X_test1
def normalize(X_train, X_test):

    X_train = X_train.drop('SK_ID_CURR',axis = 1)

    X_test = X_test.drop('SK_ID_CURR', axis = 1)

    scaler = preprocessing.StandardScaler().fit(X_train)

    print (scaler)

    xtr = pd.DataFrame((scaler.transform(X_train)).tolist())

#     xtr['TARGET'] = Y_train.values.tolist()



    xte = pd.DataFrame((scaler.transform(X_test)).tolist())



    # df_norm = shuffle(df_norm.values)

    # df_norm = pd.DataFrame(df_norm.tolist())

    return xtr, xte, 
X_train_, X_test_ = normalize (X_train, X_test)
del X_test

# Class count

df_train = X_train_

df_train['TARGET'] = Y_train.tolist()

count_class_0, count_class_1 = df_train.TARGET.value_counts()



# Divide by class

df_class_0 = df_train[df_train['TARGET'] == 0]

df_class_1 = df_train[df_train['TARGET'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)

df_undersampled = pd.concat([df_class_0_under, df_class_1], axis=0)



print('Random under-sampling:')

print(df_undersampled.TARGET.value_counts())



df_undersampled.TARGET.value_counts().plot(kind='bar', title='Count (target)');
df_undersampled_xtr = df_undersampled.drop('TARGET', axis = 1)

df_undersampled_ytr = df_undersampled['TARGET']
df_undersampled_xtr = df_undersampled.drop('TARGET', axis = 1)

df_undersampled_ytr = df_undersampled['TARGET']
del df_undersampled
X_train_.shape
def cross_validation(df_undersampled_xtr, df_undersampled_ytr, X_test, n, classifier):

    print(__doc__)



    X_tr = df_undersampled_xtr

    Y_tr = df_undersampled_ytr



    cv = StratifiedKFold(n_splits=n)

    tprs = []

    aucs = []

    mean_fpr = np.linspace(0, 1, 100)



    i = 0

    for train, test in cv.split(X_tr, Y_tr):

        probas_ = classifier.fit(X_tr.values[train], Y_tr.values[train]).predict_proba(X_tr.values[test])

        # Compute ROC curve and area the curve

        fpr, tpr, thresholds = roc_curve(Y_tr.values[test], probas_[:, 1])

        tprs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        plt.plot(fpr, tpr, lw=1, alpha=0.3,

                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))



        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

             label='Chance', alpha=.8)



    mean_tpr = np.mean(tprs, axis=0)

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)



    plt.plot(mean_fpr, mean_tpr, color='b',

             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

             lw=2, alpha=.8)



    std_tpr = np.std(tprs, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,

                     label=r'$\pm$ 1 std. dev.')



    Y_score = classifier.predict_proba(X_test.values)

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic curve')

    plt.legend(loc="lower right")

    

    plt.show()



    return Y_score
lda = LDA(n_components=1)

X_lda = lda.fit_transform(df_undersampled_xtr, df_undersampled_ytr)

X_te = lda.transform(X_test_)

Y_score = cross_validation(pd.DataFrame(X_lda.tolist()), 

                                                           pd.DataFrame(df_undersampled_ytr.tolist()), 

                                                           pd.DataFrame(X_te.tolist()), 5, 

                                      LogisticRegression(penalty='l2', dual=False, 

                                                                       tol=0.0001, C=1.0, fit_intercept=True, 

                                                                       intercept_scaling=1, class_weight=None, 

                                                                       random_state=None, solver='sag', 

                                                                       max_iter=100, multi_class='auto', 

                                                                       verbose=0, warm_start=False, 

                                                                       n_jobs=None))



result['TARGET'] = Y_score[:,1]

# result
result.to_csv('submission_logistic_reg_LDA.csv', index = False)