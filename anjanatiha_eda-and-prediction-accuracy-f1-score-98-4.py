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
import itertools

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn import metrics 

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/data.csv")

df.drop(columns=["Unnamed: 32"], inplace=True)

class_names = df.diagnosis.unique()

df.head()
df.describe()
df.info()
col = ["diagnosis"]

d = df



num_cols = d.columns[d.dtypes.apply(lambda c: np.issubdtype(c, np.number))]



scaler = StandardScaler()

d[num_cols] = scaler.fit_transform(d[num_cols])

# print(scaler.mean_)

# # scaler.transform(d)

columns = d.columns
sns.pairplot(d)
# g = sns.PairGrid(

#     d, 

#     diag_sharey=True, 

#     height=2.5, 

#     aspect=1, 

#     despine=True, 

#     dropna=False)

# g = g.map(plt.scatter)

# g.map_diag(plt.hist)

# g.map_offdiag(plt.scatter);
sns.set(style="white")

sns.clustermap(d.corr(), 

               pivot_kws=None, 

#                method='average', 

#                metric='euclidean', 

               z_score=None, 

               standard_scale=None,

               figsize=None,

               cbar_kws=None, 

               row_cluster=True, 

               col_cluster=True, 

               row_linkage=None, 

               col_linkage=None,

               row_colors=None, 

               col_colors=None, 

               mask=None,

               center=0,

               cmap="vlag",

               linewidths=.75, 

#                figsize=(13, 13)

              )
# Compute the correlation matrix

corr = d.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(len(columns), len(columns)))



# Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)

cmap = sns.diverging_palette(h_neg=220, h_pos=10, s=75, l=50, sep=10, n=len(columns), center='light', as_cmap=True)



sns.set(style="white")

sns.heatmap(corr,

         vmin=None,

         vmax=None,

         cmap=cmap,

         center=None,

         robust=True,

         annot=True, 

#          fmt='.2g',

         annot_kws=None, 

#          linewidths=0.5, 

#          linecolor='white',

         cbar=True,

         cbar_kws={"shrink": .5},

         cbar_ax=None, 

         square=True, 

         xticklabels='auto',

         yticklabels='auto', 

         mask=mask, 

         ax=None)





plt.yticks(rotation=0)

plt.xticks(rotation=90)
target_val = set(df["diagnosis"])

m = {i:v for v,i in enumerate(target_val)}

df["diagnosis"] = df["diagnosis"].map(m)
# df.dropna()

y = df["diagnosis"]

X = df.drop(columns=["id", "diagnosis"])



X = X.values

y = y.values
def print_performance(model, X_test, y_test, class_names):

    preds = model.predict(X_test)



    # accuracy_score = metrics.accuracy_score(y_test, preds)

    # auc = metrics.auc(y_test, preds)

    # average_precision_score = metrics.average_precision_score(y_test, preds)

    # balanced_accuracy_score = metrics.balanced_accuracy_score(y_test, preds)

    # brier_score_loss = metrics.brier_score_loss(y_test, preds)

    classification_report = metrics.classification_report(y_test, preds)

    # cohen_kappa_score = metrics.cohen_kappa_score(y_test, preds)

    confusion_matrix = metrics.confusion_matrix(y_test, preds)

    f1_score_ = metrics.f1_score(y_test, preds, average="weighted")

    # fbeta_score = metrics.fbeta_score(y_test, preds, average="weighted")

    # hamming_loss = metrics.hamming_loss(y_test, preds)

    # hinge_loss = metrics.hinge_loss(y_test, preds)

    # jaccard_similarity_score = metrics.jaccard_similarity_score(y_test, preds)

    # log_loss = metrics.log_loss(y_test, preds)

    # matthews_corrcoef = metrics.matthews_corrcoef(y_test, preds)

    # precision_recall_curve = metrics.precision_recall_curve(y_test, preds)

    # precision_recall_fscore_support = metrics.precision_recall_fscore_support(y_test, preds)

    # precision_score = metrics.precision_score(y_test, preds, average="weighted")

    # recall_score = metrics.recall_score(y_test, preds, average="weighted")

    # roc_auc_score = metrics.roc_auc_score(y_test, preds, average="weighted")

    # roc_curve = metrics.roc_curve(y_test, preds)

    # zero_one_loss = metrics.zero_one_loss(y_test, preds)

    

    print("-"*55)

    print("Performance")

    print("-"*55)

    # print("{} : {:.4f} ".format("Accuracy Score                  ", accuracy_score))

    # print("{} : {:.4f} ".format("AUC                             ", auc))

    # print("{} : {:.4f} ".format("Average Precision Score         ", average_precision_score))

    # print("{} : {:.4f} ".format("Balanced Accuracy Score         ", balanced_accuracy_score))

    # print("{} : {:.4f} ".format("Brier Score Loss                ", brier_score_loss))

#     print("{} : {:.4f} ".format("Classification Report           ", classification_report))

    # print("{} : {:.4f} ".format("Cohen Kappa Score               ", cohen_kappa_score))

#     print("{} : {:.4f} ".format("Confusion Matrix                ", confusion_matrix))

    print("{} : {:.4f} ".format("F1 Score                        ", f1_score_))

    # print("{} : {:.4f} ".format("Fbeta Score                     ", fbeta_score))

    # print("{} : {:.4f} ".format("Hamming Loss                    ", hamming_loss))

    # print("{} : {:.4f} ".format("Hinge Loss                      ", hinge_loss))

    # print("{} : {:.4f} ".format("Jaccard Similarity Score        ", jaccard_similarity_score))

    # print("{} : {:.4f} ".format("Log Loss                        ", log_loss))

    # print("{} : {:.4f} ".format("Matthews Corrcoef               ", matthews_corrcoef))

    # print("{} : {:.4f} ".format("Precision Recall Curve          ", precision_recall_curve))

    # print("{} : {:.4f} ".format("Precision Recall Fscore Support ", precision_recall_fscore_support))

    # print("{} : {:.4f} ".format("Precision Score                 ", precision_score))

    # print("{} : {:.4f} ".format("Recall Score                    ", recall_score))

    # print("{} : {:.4f} ".format("Roc Auc Score                   ", roc_auc_score))

    # print("{} : {:.4f} ".format("Roc Curve                       ", roc_curve))

    # print("{} : {:.4f} ".format("Zero One Loss                   ", zero_one_loss))

    print(classification_report)

    

    print("-"*55)

    print("\n\n")

    



    np.set_printoptions(precision=2)



    # Plot non-normalized confusion matrix

    plt.figure()

    plot_confusion_matrix(confusion_matrix, classes=class_names,

                          title='Confusion matrix, without normalization')



    # Plot normalized confusion matrix

    plt.figure()

    plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,

                          title='Normalized confusion matrix')



    plt.show()

    

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    

    

def print_performance_grid(clf):

    # print("*"*100)

    # print("{}{}{}".format("*"*40,"Performance", "*"*40))

    print("{}".format("Performance"))

    print("*"*90)

    print("Score            : {}".format(clf.score(X, y)))

    print("Best Estimator   : {}".format(clf.best_estimator_))

    print("Best Score       : {}".format(clf.best_score_))

    print("Best Params      : {}".format(clf.best_params_))

    print("Best Index       : {}".format(clf.best_index_))

    # print("Scorer           : {}".format(clf.scorer_))

    print("Refit Time       : {}".format(clf.refit_time_))

    # print("CV Results       : {}".format(clf.cv_results_))



    params = clf.get_params()

    best_estimator = clf.best_estimator_

    cv_results = clf.cv_results_

    

    return params, best_estimator, cv_results
# parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'), 

#               'degree': np.arrange(10),

#               'C':np.arrange(10)}



parameters = {'kernel':('linear', 'rbf'), 

              'degree': [1, 10],

              'C': [1, 10]}



svc = svm.SVC(C=1.0,

    kernel='rbf',

    degree=3, 

    gamma='auto',

    coef0=0.0,

    shrinking=True, 

    probability=False,

    tol=0.001,

    cache_size=200,

    class_weight=None, 

    verbose=False,

    max_iter=-1, 

    decision_function_shape='ovr',

    random_state=None)





svc = svm.SVC(gamma='auto')



clf = GridSearchCV(estimator=svc, 

                   param_grid=parameters,

                   scoring=None, 

                   fit_params=None, 

                   n_jobs=None,

                   iid='warn',

                   refit=True,

                   cv=5,

                   verbose=0,

                   pre_dispatch='2*n_jobs',

                   error_score='raise-deprecating',

                   return_train_score='warn')



clf.fit(X, y)

 

params, best_estimator, cv_results = print_performance_grid(clf)
parameters = {'kernel':('linear', 'rbf'), 

              'degree': np.arange(1, 10),

              'C': np.arange(1, 10)}



svc = svm.SVC(C=1.0,

    kernel='rbf',

    degree=3, 

    gamma='auto',

    coef0=0.0,

    shrinking=True, 

    probability=False,

    tol=0.001,

    cache_size=200,

    class_weight=None, 

    verbose=False,

    max_iter=-1, 

    decision_function_shape='ovr',

    random_state=None)





svc = svm.SVC(gamma='auto')



clf = GridSearchCV(estimator=svc, 

                   param_grid=parameters,

                   scoring=None, 

                   fit_params=None, 

                   n_jobs=-1,

                   iid='warn',

                   refit=True,

                   cv=5,

                   verbose=1,

                   pre_dispatch='2*n_jobs',

                   error_score='raise-deprecating',

                   return_train_score='warn')





clf.fit(X, y)



params, best_estimator, cv_results = print_performance_grid(clf)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



# svc = svm.SVC(C=1.0,

#     kernel='rbf',

#     degree=3, 

#     gamma='auto',

#     coef0=0.0,

#     shrinking=True, 

#     probability=False,

#     tol=0.001,

#     cache_size=200,

#     class_weight=None, 

#     verbose=False,

#     max_iter=-1, 

#     decision_function_shape='ovr',

#     random_state=None)



# best estimator found using grid search cv

svc = svm.SVC(C=4, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=1, gamma='auto', kernel='rbf',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)





clf = svc



print("Cross Val Score            : {}".format(cross_val_score(clf, X, y, cv=5)))



clf.fit(X_train, y_train)

print("Score (training data only) : {}".format(clf.score(X_train, y_train)))



y_pred = clf.predict(X_test)

print("F-1 Score                  : {}".format(f1_score(y_test, y_pred, average='weighted')))

      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



print("SVM")

model = clf

model.fit(X_train, y_train)

print_performance(model, X_test, y_test, class_names)