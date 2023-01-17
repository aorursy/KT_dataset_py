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
# result
dff = train_df.drop(columns = ['SK_ID_CURR', 'TARGET'], axis = 1)

feats = dff.columns.tolist()
# del train_df

# del dff
X_train1 = (train_df.drop('TARGET', axis = 1)).head(70000)

Y_train = train_df['TARGET'].head(70000)

X_test1 = (test_df.drop('TARGET', axis = 1))
# del train_df

# del test_df
from sklearn.preprocessing import Imputer

imp = Imputer(strategy='mean').fit(X_train1)

X_train = pd.DataFrame((imp.transform(X_train1)).tolist())

X_test = pd.DataFrame((imp.transform(X_test1)).tolist())
X_train.columns = X_train1.columns

X_test.columns = X_train1.columns



X_train.shape

X_train.head()
# del X_train1

# del X_test1
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
# del X_test

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
# del df_undersampled
X_train_.shape
def cor_selector(X, y,ft):

    cor_list = []

    # calculate the correlation with y for each feature

    for i in X.columns.tolist():

        cor = np.corrcoef(X[i], y)[0, 1]

        cor_list.append(abs(cor))

#     print (len(cor_list))

    # replace NaN with 0

    cor_list = [0 if np.isnan(i) else i for i in cor_list]

#     print (cor_list)

    # feature name

    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[::-1][:ft]].columns.tolist()

#     print(cor_feature)

    # feature selection? 0 for not select, 1 for select

    cor_support = [True if i in cor_feature else False for i in cor_feature]

#     print (cor_support)

    return cor_support, cor_feature
def cross_validation(feats, df_undersampled_xtr, df_undersampled_ytr, X_test, ft, n, classifier):

    print(__doc__)



    x = df_undersampled_xtr

    y = df_undersampled_ytr

    cor_support, cor_feature = cor_selector(x,y,ft)

#     print(cor_support, cor_feature)

    n_samples, n_features = x[cor_feature].shape



    # Classification and ROC analysis



    # Run classifier with cross-validation and plot ROC curves

    df_corr = df_undersampled_xtr[cor_feature]

    df_corr['TARGET'] = df_undersampled_ytr.values.tolist()

    

    fts = []

    for i in cor_feature:

        fts.append(feats[i])

        

    X_tr = df_corr.drop('TARGET',axis = 1)

    Y_tr = df_corr['TARGET']

    

#     print(cor_feature)

#     print (X_tr.columns)

#     print(fts)

    

    cv = StratifiedKFold(n_splits=n)

    tprs = []

    aucs = []

    mean_fpr = np.linspace(0, 1, 100)

    X_tr.columns = df_undersampled_xtr[cor_feature].columns

    

    

    i = 0

    feature_importance_df = pd.DataFrame()

    for train, test in cv.split(X_tr, y):

        probas_ = classifier.fit(X_tr.values[train], Y_tr.values[train]).predict_proba(X_tr.values[test])

        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = fts

        f1 = (classifier.coef_)[0].tolist()

        f = [abs(f) for f in f1]

        max_f = np.array(f).max()

        fold_importance_df["importance"] = [(100.0 * (x / max_f)) for x in f]

        fold_importance_df["fold"] = n + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



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



    Y_score = classifier.predict_proba(X_test[cor_feature].values)



    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic curve')

    plt.legend(loc="lower right")

    

    plt.show()

    

    return cor_feature, feature_importance_df, Y_score[:,1].tolist()
cor_feature, feat_imp, pred  = cross_validation(feats, df_undersampled_xtr, df_undersampled_ytr, 

                                                                              X_test_, 301, 5, 

                                          classifier =  LogisticRegression(penalty='l2', dual=False, 

                                                                           tol=0.0001, C=1e40, fit_intercept=True, 

                                                                           intercept_scaling=1, class_weight='balanced', 

                                                                           random_state=None, solver='warn', 

                                                                           max_iter=100, multi_class='warn', 

                                                                           verbose=0, warm_start=False, 

                                                                           n_jobs=None))

result['TARGET'] = pred

# result
result.to_csv('submission_logistic_reg.csv', index = False)
# Display/plot feature importance

def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", 

                                                                                                   ascending=False)[:30].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 8))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('Logistic Rgression \nFeatures (avg over folds)')

    plt.tight_layout()

    
display_importances(feat_imp)