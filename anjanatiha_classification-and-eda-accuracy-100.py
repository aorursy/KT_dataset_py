# System

import os



# Numerical

import numpy as np

import pandas as pd



# NLP

import re



# Tools

import itertools



# Machine Learning - Preprocessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# Machine Learning - Model Selection

from sklearn.model_selection import GridSearchCV





# Machine Learning - Models

from sklearn import svm



# Machine Learning - Evaluation

from sklearn import metrics 

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score



# Plot

import matplotlib.pyplot as plt

import seaborn as sns



print(os.listdir("../input"))
def get_plt_params():

    params = {'legend.fontsize': 'x-large',

              'figure.figsize' : (18, 8),

              'axes.labelsize' : 'x-large',

              'axes.titlesize' : 'x-large',

              'xtick.labelsize': 'x-large',

              'ytick.labelsize': 'x-large',

              'font.size'      :  10}

    return params
df = pd.read_csv("../input/Iris.csv")

class_names = df.Species.unique()

df.head()
df.describe()
df.info()
d = df

all_columns = d.columns

columns = list(set(all_columns) - set(["Id"]))

feature_columns = list(set(columns) - set(["Species"]))

target = "Species"
# scaler = StandardScaler()

# d = scaler.fit_transform(df) 

# d = pd.DataFrame(data=d, columns=df.columns)

# print(scaler.mean_)

# # scaler.transform(d)
figsize=(20, 8)



ticksize = 14

titlesize = ticksize + 8

labelsize = ticksize + 5



params = {'figure.figsize' : figsize,

          'axes.labelsize' : labelsize,

          'axes.titlesize' : titlesize,

          'xtick.labelsize': ticksize,

          'ytick.labelsize': ticksize}



plt.rcParams.update(params)



col = target

xlabel = "Species"

ylabel = "Count"



sns.countplot(x=df[target])

plt.title("Count of Species")

plt.xticks(rotation=90)

plt.xlabel(xlabel)

plt.ylabel(ylabel)
sns.set(style="white")



figsize=(20, 12)



ticksize = 14

titlesize = ticksize + 8

labelsize = ticksize + 5



params = {'figure.figsize' : figsize,

          'axes.labelsize' : labelsize,

          'axes.titlesize' : titlesize,

          'xtick.labelsize': ticksize,

          'ytick.labelsize': ticksize}



params = get_plt_params()

plt.rcParams.update(params)



# sns.pairplot(df[columns])



g = sns.PairGrid(df[columns])

g.map(sns.lineplot);





plt.plot()

plt.show()
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



figsize=(20, 12)



ticksize = 14

titlesize = ticksize + 8

labelsize = ticksize + 5



params = {'figure.figsize' : figsize,

          'axes.labelsize' : labelsize,

          'axes.titlesize' : titlesize,

          'xtick.labelsize': ticksize,

          'ytick.labelsize': ticksize}



params = get_plt_params()

plt.rcParams.update(params)



d = df[columns]



sns.clustermap(d.corr(), 

               figsize=(18, 12),

#                center=0,

               cmap="vlag",

              )





plt.plot()

plt.show()

sns.set(style="white")

fig = plt.figure(figsize=(18, 12))



d = df[columns]



params = get_plt_params()

plt.rcParams.update(params)





corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(len(columns)*5, len(columns)*5))



cmap = sns.diverging_palette(h_neg=220, h_pos=10, s=75, l=50, sep=10, n=len(columns), center='light', as_cmap=True)



ax = sns.heatmap(

    corr,

    cmap=cmap,

    center=0,

    robust=True,

    annot=True,

    linewidths=0.5,

    linecolor='white',

    cbar=True,

    cbar_kws={"shrink": .5},

    square=True,

    mask=mask)



# plt.yticks(rotation=0)

# plt.xticks(rotation=90)
target_val = set(df["Species"])

m = {i:v for v,i in enumerate(target_val)}

df["Species"] = df["Species"].map(m)
# df.dropna()

y = df["Species"]

X = df.drop(columns=["Id", "Species"])



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

    plt.subplot(121)

    plot_confusion_matrix(confusion_matrix, classes=class_names,

                          title='Confusion matrix, without normalization')

def print_performance(model, X_test, y_test, class_names):

    preds = model.predict(X_test)



    # accuracy_score = metrics.accuracy_score(y_test, preds)

    # auc = metrics.auc(y_test, preds)

    # average_precision_score = metrics.average_precision_score(y_test, preds)

    classification_report = metrics.classification_report(y_test, preds)

    # cohen_kappa_score = metrics.cohen_kappa_score(y_test, preds)

    confusion_matrix = metrics.confusion_matrix(y_test, preds)

    f1_score_ = metrics.f1_score(y_test, preds, average="weighted")

    # precision_recall_curve = metrics.precision_recall_curve(y_test, preds)

    # precision_score = metrics.precision_score(y_test, preds, average="weighted")

    # recall_score = metrics.recall_score(y_test, preds, average="weighted")

    # roc_auc_score = metrics.roc_auc_score(y_test, preds, average="weighted")

    # roc_curve = metrics.roc_curve(y_test, preds)

    

    print("-"*55)

    print("Performance")

    print("-"*55)

    # print("{} : {:.4f} ".format("Accuracy Score                  ", accuracy_score))

    # print("{} : {:.4f} ".format("AUC                             ", auc))

    # print("{} : {:.4f} ".format("Average Precision Score         ", average_precision_score))

#     print("{} : {:.4f} ".format("Classification Report           ", classification_report))

#     print("{} : {:.4f} ".format("Confusion Matrix                ", confusion_matrix))

    print("{} : {:.4f} ".format("F1 Score                        ", f1_score_))

    # print("{} : {:.4f} ".format("Precision Recall Curve          ", precision_recall_curve))

    # print("{} : {:.4f} ".format("Precision Score                 ", precision_score))

    # print("{} : {:.4f} ".format("Recall Score                    ", recall_score))

    # print("{} : {:.4f} ".format("Roc Auc Score                   ", roc_auc_score))

    # print("{} : {:.4f} ".format("Roc Curve                       ", roc_curve))

    print(classification_report)

    

    print("-"*55)

    print("\n\n")

    



    np.set_printoptions(precision=2)



    # Plot non-normalized confusion matrix

    plt.subplot(121)

    plot_confusion_matrix(confusion_matrix, classes=class_names, title='Confusion matrix, without normalization')



    # Plot normalized confusion matrix

    plt.subplot(122)

    plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')



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

    print("*"*100)

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
kernel = ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')

degree = np.arange(1, 10, 1)

C = np.array([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1), np.arange(1, 10, 1)]).flatten()

gamma = np.array([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1), np.arange(1, 10, 1)]).flatten()





param_grid = {

    'kernel': ('linear', 'rbf'), 

    'C': C,

    'gamma': gamma

}



estimator = svm.SVC(class_weight='balanced')



cv = 3

verbose = 0





grid_clf = GridSearchCV(estimator=estimator,param_grid=param_grid, n_jobs=-1, cv=cv, verbose=verbose)



grid_clf.fit(X, y)
params, best_estimator, cv_results = print_performance_grid(grid_clf)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)





model = svm.SVC(kernel="rbf", C=0.4, gamma=2.0, class_weight='balanced')



print("Cross Val Score            : {}".format(cross_val_score(estimator, X, y, cv=5)))



model.fit(X_train, y_train)



print("Score (training data only) : {}".format(model.score(X_train, y_train)))



y_pred = model.predict(X_test)

print("F-1 Score                  : {}".format(f1_score(y_test, y_pred, average='weighted')))

      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



print("SVM")

model = model

model.fit(X_train, y_train)

print_performance(model, X_test, y_test, class_names)